import os
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, IntervalStrategy
from transformers import SchedulerType
import argparse
from peft import LoraConfig, TaskType
import logging, datasets
import sys
from audit_llms.helpers import clean_text_qa_instruct
from trl import RewardTrainer, RewardConfig
from data import DATA_DIR

logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || " \
           f"trainable%: {100 * trainable_params / all_param}"


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='coastalcph/eu_debates', help='Dataset name')
    parser.add_argument('--party_names', default='S&D', help='List of party names to consider when filtering')
    parser.add_argument('--date_range', default=('2009-07-14', '2014-04-17'), type=tuple, help='Date range to consider when filtering')
    parser.add_argument('--min_length', default=100, help='Minimum length of the text to consider when filtering')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train')
    parser.add_argument('--per_device_train_batch_size', default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_extension', default='sd-2014', help='Output extension for output directory')
    parser.add_argument('--pseudo_qa', default=True, type=bool, help='Whether to turn the text into a pseudo question')
    parser.add_argument('--max_samples', default=20000, type=int, help='Number of samples to consider')
    parser.add_argument('--debug', default=True, type=bool, help='Whether to use debug mode')
    param_config = parser.parse_args()

    reward_config = RewardConfig(output_dir=os.path.join(DATA_DIR, 'reward_models', f'{param_config.model_name}-{param_config.output_extension}'))
    reward_config.per_device_train_batch_size = param_config.per_device_train_batch_size
    reward_config.per_device_eval_batch_size = param_config.per_device_train_batch_size
    reward_config.gradient_accumulation_steps = param_config.gradient_accumulation_steps
    reward_config.num_train_epochs = param_config.epochs
    reward_config.seed = param_config.seed
    reward_config.optim = "paged_adamw_32bit"
    reward_config.warmup_ratio = 0.05
    reward_config.weight_decay = 0.001
    reward_config.max_grad_norm = 0.3
    reward_config.learning_rate = param_config.lr
    reward_config.lr_scheduler_type = SchedulerType("constant")
    reward_config.fp16 = True if torch.cuda.is_available() else False
    reward_config.logging_strategy = IntervalStrategy("steps")
    reward_config.log_level = "info"
    reward_config.logging_first_step = True
    reward_config.save_total_limit = 5
    reward_config.logging_steps = 50
    reward_config.save_strategy = IntervalStrategy("epoch")


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if param_config.debug:
        print('Debugging mode activated')
        param_config.model_name = 'gpt2'
        tokenizer_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        param_config.quant = False
        param_config.max_length = 8
    else:
        tokenizer_name = param_config.model_name

    # Fix parties' list
    param_config.party_names = param_config.party_names.split(',') if param_config.party_names is not None else None

    # Report configuration parameters
    print('Configuration parameters:')
    for arg in vars(param_config):
        print(f'{arg}: {getattr(param_config, arg)}')

    # Compute free memory for each GPU
    if torch.cuda.is_available():
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f"{free_in_GB - 2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None

    # Load tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        param_config.model_name,
        num_labels=1,
        max_memory=max_memory,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            use_flash_attention=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        ) if param_config.debug is False else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2" if not param_config.debug else None,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare LORA model
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # Cast the output to float32
    model.score = CastOutputToFloat(model.score)

    # Set the LORA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Report the number of trainable parameters
    print(print_trainable_parameters(model))

    # Load the dataset
    dataset = load_dataset(param_config.dataset_name, split="train")

    # Filter out the samples that are not from the date range of interest
    if param_config.date_range is not None:
        dataset = dataset.filter(lambda sample: param_config.date_range[0] < sample["date"] < param_config.date_range[1])
        print('Number of samples:', len(dataset), 'from the date range of interest (', param_config.date_range, ')')

    # Filter out the samples that are too short
    if param_config.min_length is not None:
        dataset = dataset.filter(lambda sample: len(sample["text"].split(' ')) > param_config.min_length)
        print('Number of samples:', len(dataset), 'that are longer than', param_config.min_length, 'tokens')

    if param_config.pseudo_qa is not None:
        print('Turning the text into a pseudo question')
        # Turn text into a pseudo question
        dataset = dataset.map(clean_text_qa_instruct, load_from_cache_file=False)

    # Create positive and negative pairs using as reference the party of interest
    debate_title = ''
    date = ''
    debate_examples = []
    examples = {'chosen': [], 'rejected': []}
    for sample in dataset:
        if sample['debate_title'] == debate_title and sample['date'] == date:
            debate_examples.append(sample)
        else:
            if len(debate_examples) > 0:
                positives = [example for example in debate_examples if example["speaker_party"] in param_config.party_names]
                negatives = [example for example in debate_examples if example["speaker_party"] not in param_config.party_names]
                if len(positives) > 0 and len(negatives) > 0:
                    for positive in positives:
                        for negative in negatives:
                            examples['chosen'].append(positive)
                            examples['rejected'].append(negative)
            debate_title = sample['debate_title']
            date = sample['date']
            debate_examples = []

    # Turn dict into HF dataset
    dataset = datasets.Dataset.from_dict(examples)
    # Subsample the dataset
    dataset = dataset.shuffle(seed=param_config.seed).select(range(min(param_config.max_samples, len(dataset))))

    # Tokenize the dataset

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen['text'])
            tokenized_rejected = tokenizer(rejected['text'])

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
    )

    # Split the dataset
    dataset = dataset.train_test_split(test_size=0.1)

    # Prepare the dataset for training
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=lora_config,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == '__main__':
    main()