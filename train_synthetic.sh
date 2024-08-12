#!/usr/bin/env bash
DTU_TRAINING="/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/"
DTU_TRAINLIST="/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/synthetic_pairs.txt"
DTU_TESTLIST="/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan/synthetic_pairs.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/snowmvps_stage4/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
CUDA_VISIBLE_DEVICES=0,1
DTU_CKPT_FILE=$DTU_LOG_DIR"/finalmodel.ckpt"
DTU_OUT_DIR="./outputs/snowmvps_stage4/"$exp



# python -m torch.distributed.launch --nproc_per_node=1 train_mvps.py --logdir $DTU_LOG_DIR --dataset=general_eval4_synthetic_ps --batch_size=16 --trainpath=$DTU_TRAINING --summary_freq 100 \
#                 --group_cor --rt --inverse_depth --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt

# python train_mvps.py --logdir $DTU_LOG_DIR --dataset=general_eval4_synthetic_ps --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 --loadckpt="./checkpoints/mvps_resume/3.ckpt" \
#                 --group_cor --rt --inverse_depth --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt

python train_mvps_stage4.py --logdir $DTU_LOG_DIR --dataset=general_eval4_synthetic_ps_stage4 --batch_size=5 --trainpath=$DTU_TRAINING --summary_freq 100 \
                --group_cor --rt --inverse_depth --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST  $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
