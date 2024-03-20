#!/bin/bash
#SBATCH --cpus-per-task=8 --mem=16000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=setting_a.txt
#SBATCH --time=1:00:00

#. /etc/profile.d/modules.sh
#eval "$(conda shell.bash hook)"
#conda activate peft

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

# Model Parameters
MODEL_PATH='meta-llama/Llama-2-13b-chat-hf'

export PYTHONPATH=.

python ./audit_llms/setting_a.py \
  --model_name ${MODEL_PATH} \
  --quant false \
  --repetition_penalty 1.0 \
  --max_length 128