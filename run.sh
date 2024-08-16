#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=iris5
#SBATCH --cpus-per-task=24
#SBATCH --job-name="threading the needle"
#SBATCH --time=3-0:0
#SBATCH --account=iris

export TASK_NAME=${1}
export SEED=${2}
export BATCH_SIZE=${3}
export POLICY_CLASS=${4}
export DT=$(date '+%d_%m_%Y_%H_%M_%S')

source /sailhome/zachwitz/.bashrc
conda deactivate
conda activate act_conda_env
cd /iris/u/zachwitz/act-plus-plus/

export MUJOCO_GL=egl
python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir /iris/u/zachwitz/act-plus-plus/ckpt/${POLICY_CLASS}_${TASK_NAME}_${SEED}_${DT} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} --dim_feedforward 3200 \
    --seed ${SEED} --num_steps 200000  --lr 1e-6 \
    --logging_mode online --fine_tune_last_layers \
    --load_pretrain /iris/u/kayburns/threading_the_needle/act-plus-plus/ckpt/ACT_aloha_thread_blue_needle_glue_1_04_06_2024_13_51_46/policy_step_200000_seed_1.ckpt

    # --seed ${SEED} --num_steps 2000000  --lr 1e-7 \
    # --logging_mode online --fine_tune_last_layers \
    # --load_pretrain /iris/u/kayburns/threading_the_needle/act-plus-plus/ckpt/ACT_aloha_thread_blue_needle_glue_1_04_06_2024_13_51_46/policy_step_200000_seed_1.ckpt
