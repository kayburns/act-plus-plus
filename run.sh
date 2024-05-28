#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=iris[1-4,8-10],iris-hp-z8
#SBATCH --job-name="threading the needle"
#SBATCH --time=3-0:0
#SBATCH --account=iris

export TASK_NAME=${1}
export SEED=${2}
export NUM_EPISODES=${3}
export BATCH_SIZE=${4}
export POLICY_CLASS=${5}
export LOSS_FN=${6}
export DT=$(date '+%d_%m_%Y_%H_%M_%S')

#source /sailhome/kayburns/.bashrc
#conda deactivate
#conda activate dev
cd /iris/u/kylehsu/code/threading-the-needle/act-plus-plus/

export DATA_DIR='/scr/optometrists/'

export MUJOCO_GL=egl
python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir /iris/u/kylehsu/code/threading-the-needle/act-plus-plus/ckpt/${POLICY_CLASS}_${TASK_NAME}_${SEED}_${DT}_${LOSS_FN} \
    --policy_class ${POLICY_CLASS} \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} --dim_feedforward 3200 \
    --seed ${SEED} --num_steps 200000  --lr 1e-5 \
    --logging_mode online --loss_fn ${LOSS_FN}
