#!/usr/bin/env bash
DTU_TESTPATH="/ssd3/hsy/Dataset/DiLiGenT-MV/mvpmsData"
DTU_TESTLIST="lists/diligent_mv/test.txt"

exp=$1

DTU_LOG_DIR="./checkpoints/mvps/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_CKPT_FILE=$DTU_LOG_DIR"1.ckpt"
DTU_OUT_DIR="./outputs/diligent_mv/"$exp


python test_diligent.py --dataset=general_eval4_diligent --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
             --inverse_depth --group_cor | tee -a $DTU_LOG_DIR/log_test.txt
