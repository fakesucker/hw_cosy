source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2

cd /home/work_nfs23/hkxie/hw_proj/CosyVoice
# RUN_MODE=serial \

RUN_MODE=parallel \
GPU_LIST=0,1,2,3,4,5,6,7 \
PROCS_PER_GPU=1 \
SFT_SPK_ID="中文女" \
META_FILE=/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu_0421_onlymale.lst \
bash infer_shenhu_sft_ckpt_sweep_spk.sh


RUN_MODE=parallel \
GPU_LIST=0,1,2,3,4,5,6,7 \
PROCS_PER_GPU=1 \
SFT_SPK_ID="中文女" \
META_FILE=/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu_0421_onlyhw_niren.lst \
bash infer_shenhu_sft_ckpt_sweep_spk.sh