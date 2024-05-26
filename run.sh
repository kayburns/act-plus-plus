#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=iris5
#SBATCH --cpus-per-task=16
#SBATCH --job-name="threading the needle"
#SBATCH --time=3-0:0
#SBATCH --account=iris

export TASK_NAME=${1}
export SEED=${2}
export NUM_EPISODES=${3}
export BATCH_SIZE=${4}
export POLICY_CLASS=${5}
export DT=$(date '+%d_%m_%Y_%H_%M_%S')

source /sailhome/kayburns/.bashrc
conda deactivate
conda activate dev
cd /iris/u/kayburns/threading_the_needle/act-plus-plus/

export MUJOCO_GL=egl
python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir /iris/u/kayburns/threading_the_needle/act-plus-plus/ckpt/${POLICY_CLASS}_${TASK_NAME}_${SEED}_${DT} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} --dim_feedforward 3200 \
    --seed ${SEED} --num_steps 200000  --lr 1e-5 \
    --logging_mode online
