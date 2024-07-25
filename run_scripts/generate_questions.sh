#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/eu-politics-llms-chronos/generate_questions.txt
#SBATCH --time=10:00:00

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate peft

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

# Model Parameters
MODEL_PATH='meta-llama/Meta-Llama-3.1-8B-Instruct'

export PYTHONPATH=.

python ./augment_data/generate_questions.py \
  --model_name ${MODEL_PATH} \
  --max_length 64 \
  --debug false
