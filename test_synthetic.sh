#!/usr/bin/env bash
DTU_TESTPATH="/ssd3/hsy/Dataset/MVPS_dataset_2022b_vulcan"
DTU_TESTLIST="lists/synthetic_ps/test.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/mvps/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"/3.ckpt"
DTU_OUT_DIR="./outputs/synthetic_ps/"$exp


python test_synthetic.py --dataset=general_eval4_synthetic_ps --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --inverse_depth --group_cor | tee -a $DTU_LOG_DIR/log_test.txt
