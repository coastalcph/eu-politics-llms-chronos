#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/eu-politics-llms-chronos/finetune_mistral_id_2019.txt
#SBATCH --time=6:00:00

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate peft

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

# Model Parameters
MODEL_PATH='mistralai/Mistral-7B-Instruct-v0.2'

export PYTHONPATH=.

python ./finetune_llms/finetune_llms.py \
  --model_name ${MODEL_PATH} \
  --party_names 'ID' \
  --legislature '9th' \
  --output_extension 'id-2019'