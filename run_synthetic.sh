export CUDA_VISIBLE_DEVICES="4"

path_dataset=/data0/datasets/nerf/nerf_synthetic
scene=lego
path_source="$path_dataset"/"$scene"

declare -a sh_mask_lambda_list=(
    0.0005
    0.001
    0.0025
    0.005
    0.01
    0.025
)
declare -a gs_mask_lambda_list=(
    0.0001
    0.0002
    0.0005
    0.001
    0.002
    0.005
)

for ((i = 0; i < ${#gs_mask_lambda_list[@]}; i++)); do
    path_output=output/nerf_synthetic/"$scene"/rate${i}
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
        --vq_scale_cbsize 4096 \
        --vq_rot_cbsize 4096 \
        --vq_dc_cbsize 4096 \
        --vq_sh1_cbsize 2048 \
        --vq_sh2_cbsize 2048 \
        --vq_sh3_cbsize 2048 \
        --vq_patch_size 65536 \
        --sh_mask_lambda ${sh_mask_lambda_list[i]} \
        --sh_mask_lr 0.005 \
        --gs_mask_lambda ${gs_mask_lambda_list[i]} \
        --gs_mask_lr 0.01 \
        --eval
    python render.py -m $path_output -s $path_source --skip_train
    python metrics.py -m $path_output
done