#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/eu-politics-llms-chronos/finetune_llama_3_id_2019.txt
#SBATCH --time=6:00:00

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate peft

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

# Model Parameters
MODEL_PATH='meta-llama/Meta-Llama-3.1-8B-Instruct'

export PYTHONPATH=.

python ./finetune_llms/finetune_llms.py \
  --model_name ${MODEL_PATH} \
  --party_names 'ID' \
  --legislature '9th' \
  --output_extension 'id-2019-v4'