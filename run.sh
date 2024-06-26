#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=iris8
#SBATCH --cpus-per-task=24
#SBATCH --job-name="threading the needle"
#SBATCH --time=3-0:0
#SBATCH --account=iris

export TASK_NAME=${1}
export SEED=${2}
export BATCH_SIZE=${3}
export POLICY_CLASS=${4}
export DT=$(date '+%d_%m_%Y_%H_%M_%S')

#source /sailhome/kayburns/.bashrc
#conda deactivate
#conda activate dev
#cd /iris/u/kayburns/threading_the_needle/act-plus-plus/

#export MUJOCO_GL=egl
#    --ckpt_dir /iris/u/kayburns/threading_the_needle/act-plus-plus/ckpt/${POLICY_CLASS}_${TASK_NAME}_${SEED}_${DT} \

python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ./ckpt_dir_test \
    --resume_ckpt_path ./policy_step_150000_seed_1.ckpt \
    --policy_class ${POLICY_CLASS} \
    --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} --dim_feedforward 3200 \
    --seed ${SEED} --num_steps 200000  --lr 1e-5 \
    --eval