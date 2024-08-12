#!/usr/bin/env bash
DTU_TESTPATH="/ssd3/hsy/Dataset/DiLiGenT-MV/mvpmsData"
DTU_TESTLIST="lists/diligent_mv/test.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/snowmvps_stage4/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

DTU_CKPT_FILE=$DTU_LOG_DIR"0.ckpt"
DTU_OUT_DIR="./outputs/diligent_snowmvps_stage4/"$exp

CUDA_VISIBLE_DEVICES=2

python test_diligent_stage4.py --dataset=general_eval4_diligent_stage4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --inverse_depth --group_cor | tee -a $DTU_LOG_DIR/log_test.txt
