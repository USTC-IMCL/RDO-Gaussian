export CUDA_VISIBLE_DEVICES="5"

path_dataset=/data0/datasets/nerf/tandt
scene=truck
path_source="$path_dataset"/"$scene"

declare -a sh_mask_lambda_list=(
    0.005
    0.02
    0.05
    0.1
    0.2
    0.5
)
declare -a gs_mask_lambda_list=(
    0.0005
    0.002
    0.005
    0.01
    0.02
    0.05
)

for ((i = 0; i < ${#gs_mask_lambda_list[@]}; i++)); do
    path_output=output/tandt/"$scene"/rate${i}
    mkdir -p $path_output
    python -u train.py \
        -s="$path_source" \
        -m="$path_output" \
        --iterations 30000 \
        --vq_cb_lr 0.0002 \
        --vq_logits_lr 0.002 \
        --vq_scale_lmbda 32768 \
        --vq_rot_lmbda 256 \
        --vq_dc_lmbda 256 \
        --vq_sh1_lmbda 256 \
        --vq_sh2_lmbda 256 \
        --vq_sh3_lmbda 256 \
        --vq_scale_cbsize 8192 \
        --vq_rot_cbsize 8192 \
        --vq_dc_cbsize 8192 \
        --vq_sh1_cbsize 4096 \
        --vq_sh2_cbsize 4096 \
        --vq_sh3_cbsize 4096 \
        --vq_patch_size 65536 \
        --sh_mask_lambda ${sh_mask_lambda_list[i]} \
        --sh_mask_lr 0.05 \
        --gs_mask_lambda ${gs_mask_lambda_list[i]} \
        --gs_mask_lr 0.01 \
        --eval
    python render.py -m $path_output -s $path_source --skip_train
    python metrics.py -m $path_output
done