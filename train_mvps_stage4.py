#!/usr/bin/env python
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from models.SDPS_Net4 import NENet
from utils.utils import *
import torch.distributed as dist

cudnn.benchmark = True
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSTER')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='general_eval4_synthetic_ps', help='select dataset')
parser.add_argument('--trainpath', default='/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default='/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/synthetic_pairs.txt', help='train list')
parser.add_argument('--testlist', default='/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/synthetic_pairs.txt', help='test list')

parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--lrepochs', type=str, default="4,8,11,13,16,18,19:1.5", help='epoch ids to downscale lr and the downscale rate')

parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='checkpoints/logs', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', default=False, action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=2, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', default=False, action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=2)

#PS Model
parser.add_argument('--ps_fuse_type', default='max', type=str)
parser.add_argument('--ps_feat_chs', type=int, default=16)
parser.add_argument('--ps_loadckpt', default='/ssd3/hsy/SnowMVPS/checkpoints/mvpsnet_pretrained.ckpt')

# MVS model
parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
# parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="1,1,1,1", help='depth loss weight for different stage')

parser.add_argument('--l1ce_lw', type=str, default="0,1", help='loss weight for l1 and ce loss')
parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")

parser.add_argument('--group_cor', default=True, action='store_true',help='group correlation')
# parser.add_argument('--group_cor_dim', type=str, default="64,32,16", help='group correlation dim')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')

parser.add_argument('--inverse_depth', default=True, action='store_true',help='inverse depth')
parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--dcn', default=False, action='store_true',help='dcn')
parser.add_argument('--pos_enc', type=int, default=0, help='pos_enc: 0 no pos enc; 1 depth sine; 2 learnable pos enc')
parser.add_argument('--arch_mode', type=str, default="fpn")

parser.add_argument('--ot_continous', default=False, action='store_true',help='optimal transport continous gt bin')
parser.add_argument('--ot_iter', type=int, default=10)
parser.add_argument('--ot_eps', type=float, default=1)

parser.add_argument('--rt', default=True, action='store_true',help='robust training')

parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--use_raw_train', default=False, action='store_true',help='using 1200x1600 training')
parser.add_argument('--mono', default=False, action='store_true',help='query to build mono depth prediction and loss')
parser.add_argument('--lr_scheduler', type=str, default='MS')
parser.add_argument('--ASFF', default=False, action='store_true')
parser.add_argument('--attn_temp', type=float, default=2)

parser.add_argument('--n_views', type=float, default=3)

os.environ['LOCAL_RANK'] = '0'
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('num gpus :',num_gpus)
is_distributed = num_gpus > 1


# main function
def train(model, ps_model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    if args.lr_scheduler == 'MS':
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                            last_epoch=len(TrainImgLoader) * start_epoch - 1)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs*len(TrainImgLoader)), eta_min=0)
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,total_steps=int(args.epochs*len(TrainImgLoader)))

    for epoch_idx in range(start_epoch, args.epochs):
        global_step = len(TrainImgLoader) * epoch_idx

        progress_bar = tqdm(total=len(TrainImgLoader))
        progress_bar.set_description(f"Epoch: {epoch_idx}")

        ### training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, ps_model, model_loss, optimizer, sample, args)
            lr_scheduler.step()

            d_loss = (scalar_outputs["s0_d_loss"] + scalar_outputs["s1_d_loss"] + scalar_outputs["s2_d_loss"] + scalar_outputs["s3_d_loss"]) / 4
            n_loss = scalar_outputs["n_loss"]
            d_error = scalar_outputs["abs_depth_error"]
            # range_error = (scalar_outputs["s0_range_err_ratio"] + scalar_outputs["s1_range_err_ratio"] + scalar_outputs["s2_range_err_ratio"]) / 3

            logs = {'Epoch': epoch_idx, 'Iter': batch_idx, 'loss': loss, 'd_loss': d_loss, 'n_loss': n_loss, 'd_error': d_error}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    if not (len(image_outputs['ref_img'].shape) == 3 or len(image_outputs['ref_img'].shape) == 4):
                        image_outputs['ref_img'] = image_outputs['ref_img'][:,0,:,:,:]
                    save_images(logger, 'train', image_outputs, global_step)

                del scalar_outputs, image_outputs

        progress_bar.close()

        ### checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                if epoch_idx == args.epochs - 1:
                    torch.save({
                        'epoch': epoch_idx,
                        'ps_model': ps_model.module.state_dict(),
                        'mvs_model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        "{}/finalmodel.ckpt".format(args.logdir))
                else:
                    torch.save({
                        'epoch': epoch_idx,
                        'ps_model': ps_model.module.state_dict(),
                        'mvs_model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        "{}/".format(args.logdir)+str(epoch_idx)+".ckpt")
                    
        gc.collect()

        ### testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, ps_model, model_loss, sample, args)

                d_loss = (scalar_outputs["s0_d_loss"] + scalar_outputs["s1_d_loss"] + scalar_outputs["s2_d_loss"] + scalar_outputs["s3_d_loss"]) / 4
                n_loss = scalar_outputs["n_loss"]
                d_error = scalar_outputs["abs_depth_error"]
                # range_error = (scalar_outputs["s0_range_err_ratio"] + scalar_outputs["s1_range_err_ratio"] + scalar_outputs["s2_range_err_ratio"]) / 3

                logs = {'Epoch': epoch_idx, 'Iter': batch_idx, 'loss': loss, 'd_loss': d_loss, 'n_loss': d_error, 'range_error': d_error}
                
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        if not (len(image_outputs['ref_img'].shape) == 3 or len(image_outputs['ref_img'].shape) == 4):
                            image_outputs['ref_img'] = image_outputs['ref_img'][:,0,:,:,:]
                        save_images(logger, 'test', image_outputs, global_step)
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, ps_model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, ps_model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, ps_model, model_loss, optimizer, sample, args):
    model.train()
    ps_model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"] # all views

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)] # all views
    mask_ref = mask[:, 0] # reference view

    ps_outputs = {}
    feats, ps_outputs['normal'] = ps_model(sample_cuda)
    normal_est = ps_outputs['normal'] # all views
    
    outputs = model(sample_cuda["imgs"], feats, sample_cuda["R"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], normal_est[:, 0])
    depth_est = outputs["depth"]

    loss, stage_d_loss, normal_loss, range_err_ratio = model_loss(sample_cuda["depth_values"], outputs, normal_est,
                                        depth_gt_ms, mask_ms, sample_cuda['normals']['stage4'],
                                        stage_lw=[float(e) for e in args.dlossw.split(",") if e], 
                                        l1ce_lw=[float(lw) for lw in args.l1ce_lw.split(",")],
                                        inverse_depth=args.inverse_depth,
                                        ot_iter=args.ot_iter, ot_continous=args.ot_continous, ot_eps=args.ot_eps,
                                        mono=args.mono
                                        )
    loss.backward()
    optimizer.step()

    d_min = sample_cuda["depth_values"].min()
    d_max = sample_cuda["depth_values"].max()
    depth_est_nomask_nor = ((depth_est).clip(d_min, d_max) - d_min) / (d_max - d_min)
    depth_est_nor = ((depth_est * mask_ref).clip(d_min, d_max) - d_min) / (d_max - d_min)
    depth_gt_nor = ((depth_gt * mask_ref).clip(d_min, d_max) - d_min) / (d_max - d_min)

    scalar_outputs = {"loss": loss,
                      "s0_d_loss": stage_d_loss[0],
                      "s1_d_loss": stage_d_loss[1],
                      "s2_d_loss": stage_d_loss[2],
                      "s3_d_loss": stage_d_loss[3],
                      "s0_range_err_ratio":range_err_ratio[0],
                      "s1_range_err_ratio":range_err_ratio[1],
                      "s2_range_err_ratio":range_err_ratio[2],
                      "s3_range_err_ratio":range_err_ratio[3],
                      "n_loss": normal_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask_ref > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask_ref > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask_ref > 0.5, 4),
                      }

    image_outputs = {"depth_est": depth_est_nor,
                    #  "depth_est_nomask": depth_est_nomask_nor,
                     "depth_gt": depth_gt_nor,
                     "normal_gt": sample['normals']['stage4'][:, 0], # reference view
                     "normal_est": normal_est[:, 0] * mask_ref.unsqueeze(1), # reference view
                     "ref_img": sample["imgs"][0],
                     "mask": mask_ref,
                     "errormap": (depth_est_nor - depth_gt_nor).abs(),
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, ps_model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
        ps_model_eval = ps_model.module
    else:
        model_eval = model
        ps_model_eval = ps_model
    model_eval.eval()
    ps_model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)][:, 0] # reference view

    ps_outputs = {}
    feats, ps_outputs['normal'] = ps_model(sample_cuda)
    normal_est = ps_outputs['normal'] # all views
    
    outputs = model(sample_cuda["imgs"], feats, sample_cuda["R"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], normal_est[:, 0])
    depth_est = outputs["depth"]
    #depth_reg = outputs["depth_reg"]
    num_valid_volume = outputs["valid_volume"]
    valid_vol_mask = torch.mean(num_valid_volume, dim = 1)/4.0
    re_mask = valid_vol_mask > 0.1
    mask = mask > 0.5
    fmask = mask*re_mask
    
    loss, stage_d_loss, normal_loss, range_err_ratio = model_loss(sample_cuda["depth_values"], outputs, normal_est,
                                        depth_gt_ms, mask_ms, sample_cuda['normals']['stage4'],
                                        stage_lw=[float(e) for e in args.dlossw.split(",") if e], 
                                        l1ce_lw=[float(lw) for lw in args.l1ce_lw.split(",")],
                                        inverse_depth=args.inverse_depth,
                                        ot_iter=args.ot_iter, ot_continous=args.ot_continous, ot_eps=args.ot_eps,
                                        mono=args.mono
                                        )

    d_min = sample_cuda["depth_values"].min()
    d_max = sample_cuda["depth_values"].max()
    depth_est_nomask_nor = ((depth_est).clip(d_min, d_max) - d_min) / (d_max - d_min)
    depth_est_nor = ((depth_est * mask).clip(d_min, d_max) - d_min) / (d_max - d_min)
    depth_gt_nor = ((depth_gt * mask).clip(d_min, d_max) - d_min) / (d_max - d_min)

    scalar_outputs = {"loss": loss,
                      "s0_d_loss": stage_d_loss[0],
                      "s1_d_loss": stage_d_loss[1],
                      "s2_d_loss": stage_d_loss[2],
                      "s3_d_loss": stage_d_loss[3],
                      "s0_range_err_ratio":range_err_ratio[0],
                      "s1_range_err_ratio":range_err_ratio[1],
                      "s2_range_err_ratio":range_err_ratio[2],
                      "s3_range_err_ratio":range_err_ratio[3],
                      "n_loss": normal_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      }

    image_outputs = {"depth_est": depth_est_nor,
                    #  "depth_est_nomask": depth_est_nomask_nor,
                     "depth_gt": depth_gt_nor,
                     "normal_gt": sample['normals']['stage3'][:, 0], # reference view
                     "normal_est": normal_est[:, 0] * mask.unsqueeze(1), # reference view
                     "ref_img": sample["imgs"][0],
                     "mask": mask,
                     "errormap": (depth_est_nor - depth_gt_nor).abs(),
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None

    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # Light Encoding Model
    ps_model = NENet(base_chs=args.ps_feat_chs, fuse_type=args.ps_fuse_type, c_in=6)
    ps_model.to(device)
    
    # model, optimizer
    model = MVS4net(arch_mode=args.arch_mode, reg_net=args.reg_mode, num_stage=4, 
                    fpn_base_channel=args.fpn_base_channel, reg_channel=args.reg_channel, 
                    stage_splits=[int(n) for n in args.ndepths.split(",")], 
                    depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
                    group_cor=args.group_cor, group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
                    inverse_depth=args.inverse_depth,
                    agg_type=args.agg_type,
                    dcn=args.dcn,
                    pos_enc=args.pos_enc,
                    mono=args.mono,
                    asff=args.ASFF,
                    attn_temp=args.attn_temp,
                )

    model.to(device)
    model_loss = MVS4net_loss

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    optimizer = optim.Adam(list(model.parameters()) + list(ps_model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    
    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        #saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['mvs_model'])
        ps_model.load_state_dict(state_dict['ps_model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    if is_distributed:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )
        ps_model = torch.nn.parallel.DistributedDataParallel(
            ps_model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            ps_model = nn.DataParallel(ps_model)
            
    # # load pretrained ps_model
    if args.loadckpt is None:
        ps_model_param = torch.load(args.ps_loadckpt)
        ps_model.load_state_dict(ps_model_param['ps_model'])

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    if args.dataset.startswith('dtu'):
        train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, args.interval_scale, rt=args.rt,  use_raw_train=args.use_raw_train)
        test_dataset = MVSDataset(args.testpath, args.testlist, "val", 5, args.interval_scale)
    elif args.dataset.startswith('blendedmvs'):
        train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 9, robust_train=args.rt)
        test_dataset = MVSDataset(args.testpath, args.testlist, "val", 9)
    elif args.dataset.startswith('general_eval4_synthetic_ps'):
        train_dataset = MVSDataset(args.trainpath, "train", args.n_views, add_domains_to_load=['light_dir', 'normal']) #root_dir, split, nviews
        test_dataset = MVSDataset(args.testpath, "test", args.n_views, add_domains_to_load=['light_dir', 'normal'])
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, ps_model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, ps_model, model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError