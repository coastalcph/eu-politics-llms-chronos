from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
from helpers import normalize_responses
from configure_prompt import build_prompt
from helpers import PROMPTS
from data import DATA_DIR
from peft import PeftModel
import argparse
import json


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model name in HF Hub')
    parser.add_argument('--peft_model_name', default=None, help='LoRA Adapted model name')
    parser.add_argument('--party', default='S&D', help='Party name to consider when filtering')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='Repetition penalty')
    parser.add_argument('--max_length', default=128, type=int, help='Maximum length of the generated text')
    parser.add_argument('--debug', default=True, type=bool, help='Whether to use debug mode')
    config = parser.parse_args()

    if config.debug:
        print('Debugging mode activated')
        config.model_name = 'gpt2'
        tokenizer_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        config.peft_model_name = None
        config.max_length = 8
    else:
        tokenizer_name = config.model_name

    # Load EUANDI questionnaire dataset
    euandi_questionnaire = []
    with open(os.path.join(DATA_DIR, 'euandi_2024', 'euandi_2024_questionnaire.jsonl'), 'r') as f_q:
        for line in f_q:
            euandi_questionnaire.append(json.loads(line))
    for idx, example in enumerate(euandi_questionnaire):
        euandi_questionnaire[idx] = build_prompt(example)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Compute free memory for each GPU
    if torch.cuda.is_available():
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f"{free_in_GB - 2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None

    if config.peft_model_name is None:
        print('Loading model from HF Hub...')
        output_name = config.model_name.split('/')[-1]
        print('Quantizing model...')
        model_config = transformers.AutoConfig.from_pretrained(
            config.model_name,
            token=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config= transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                use_flash_attention=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            ) if config.debug is False else None,
            device_map='auto',
            token=True,
            max_memory=max_memory if not config.debug else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            attn_implementation="flash_attention_2" if not config.debug else None,
        )
    else:
        print('Loading custom DAPT model locally..')
        output_name = config.peft_model_name.split('/')[-1]
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,
                                                                  quantization_config=transformers.BitsAndBytesConfig(
                                                                    load_in_4bit=True,
                                                                    use_flash_attention=True,
                                                                    bnb_4bit_compute_dtype=torch.float16,
                                                                    bnb_4bit_use_double_quant=False,
                                                                    bnb_4bit_quant_type="nf4",
                                                                ) if config.debug is False else None,
                                                                  device_map="auto" if torch.cuda.is_available() else "cpu",
                                                                  torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                                                                  max_memory=max_memory)
        model = PeftModel.from_pretrained(model, config.peft_model_name,
                                          device_map="auto",
                                          max_memory=max_memory if not config.debug else None)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Iterate over the examples in the dataset and save the responses
    examples = []
    for example in tqdm.tqdm(euandi_questionnaire):
        # Print the instruction
        print('INSTRUCTION:\n', example["annotation_request"])
        for idx, system_prompt in enumerate(PROMPTS):
            temp_prompt = system_prompt.format(config.party)
            annotation_request = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": temp_prompt},
                                                                             {"role": "user", "content": example["annotation_request"]}],
                                                               tokenize=False, add_generation_prompt=False)

            annotation_request += 'I am most aligned with option ('
            try:
                # Get the response from the chatbot
                responses = pipeline(
                    annotation_request,
                    do_sample=True,
                    num_return_sequences=1,
                    return_full_text=False,
                    max_new_tokens=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    repetition_penalty=config.repetition_penalty,
                )

                # Print the response
                print(f'RESPONSE GIVEN PROMPT [{idx}]:\nI am most aligned with option ({responses[0]["generated_text"].strip()}')
                print("-" * 50)
                # Save the response
                example[f"response_{idx}"] = '(' + responses[0]['generated_text'].strip()
            except:
                print('RESPONSE: None\n')
                # Save the response
                example[f"response_{idx}"] = 'N/A'
                examples.append(example)
        examples.append(example)

    # Print statistics
    print("Number of examples:", len(examples))

    # Normalize the responses
    for idx in range(len(PROMPTS)):
        examples = normalize_responses(examples, idx, config.shuffle)

    # Save the responses to a jsonl file
    with open(os.path.join(DATA_DIR, "model_responses", f"{output_name}_responses.jsonl"), "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


if __name__ == '__main__':
    main()