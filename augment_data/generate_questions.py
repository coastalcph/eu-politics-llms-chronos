from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
from data import DATA_DIR
from typing import List
import argparse
import json
from nltk.tokenize import sent_tokenize, word_tokenize


SYSTEM_PROMPT = 'You are a helpful AI assistant with expertise on EU politics.'
INSTRUCTION = 'This is a statement by an MEP of the {} political group in the European Parliament: "{}".\n\n What do you think was the question asked? The question should start as "What is your opinion on..." and should not be longer than 16 words.'
ASSISTANT_START = 'The question asked was likely: "What is your opinion on '

def truncate_text(text, max_length):
    ''' Truncate text to the maximum length '''
    sentences = sent_tokenize(text)
    truncated_text = ''
    truncated_text_length = 0
    for sentence in sentences:
        sentence_length = len(word_tokenize(sentence))
        if sentence_length + truncated_text_length < max_length:
            truncated_text += sentence
            truncated_text_length += sentence_length
        else:
            break
    return truncated_text

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model name in HF Hub')
    parser.add_argument('--max_length', default=64, type=int, help='Maximum length of the generated text')
    parser.add_argument('--parties', default=['PPE', 'S&D', 'GUE/NGL'], type=List, help='List of party names to consider when filtering')
    parser.add_argument('--debug', default=True, type=bool, help='Whether to use debug mode')
    config = parser.parse_args()

    # Load eu-elections dataset
    dataset = load_dataset('coastalcph/eu_debates', split='train')

    if config.debug:
        print('Debugging mode activated')
        config.model_name = 'gpt2'
        config.quant = False
        config.max_length = 8

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=True)

    # Compute free memory for each GPU
    if torch.cuda.is_available():
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f"{free_in_GB - 2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None

    # Compute free memory for each GPU
    print('Loading model from HF Hub...')
    bnb_config = None
    model_config = transformers.AutoConfig.from_pretrained(
        config.model_name,
        token=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        token=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        max_memory=max_memory
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Iterate over the examples in the dataset and save the responses
    examples = 0
    with open(os.path.join(DATA_DIR, 'eu_parliaments_extended.json'), 'w') as f:
        for example in tqdm.tqdm(dataset):
            if example['speaker_party'] not in config.parties:
                continue
            try:
                # Truncate the text to the maximum length
                if len(example['text'].split(' ')) < 100:
                    continue
                elif len(example['text'].split(' ')) > 256:
                    truncated_text = truncate_text(example['text'], 256)
                else:
                    truncated_text = example['text']

                # Print the instruction
                annotation_request = tokenizer.apply_chat_template(
                    conversation=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": INSTRUCTION.format(example['speaker_party'], truncated_text.strip())}],
                    tokenize=False, add_generation_prompt=False)
                annotation_request += ASSISTANT_START
                print('INSTRUCTION:\n', annotation_request.split('user<|end_header_id|>\n\n ')[1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip())
                # Get the response from the chatbot
                responses = pipeline(
                    annotation_request,
                    do_sample=True,
                    num_return_sequences=1,
                    return_full_text=False,
                    max_new_tokens=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
                # Print the response
                print(
                    f'RESPONSE:\nThe question asked was likely: "What is your opinion on {responses[0]["generated_text"].strip()}')
                print("-" * 50)
                # Save the response
                example[
                    "question"] = f'"What is your opinion on {responses[0]["generated_text"].strip()}'
                f.write(json.dumps(example) + '\n')
            except:
                print('ERROR: Could not generate question for example: ', example['text'])
                print("-" * 50)
                example["question"] = None
                f.write(json.dumps(example) + '\n')
                continue
            examples += 1

    # Print statistics
    print("Number of examples:", len(examples))


if __name__ == '__main__':
    main()
