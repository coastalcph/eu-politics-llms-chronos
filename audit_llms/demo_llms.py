from transformers import AutoTokenizer
import transformers
import torch
from peft import PeftModel


model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
peft_model_path = input('Peft Model Path: ')
party_name = input('Party Name: ')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Loading custom DAPT model locally..')
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                          quantization_config=transformers.BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            use_flash_attention=True,
                                                            bnb_4bit_compute_dtype=torch.float16,
                                                            bnb_4bit_use_double_quant=False,
                                                            bnb_4bit_quant_type="nf4",
                                                        ) if torch.cuda.is_available() else None,
                                                          device_map="auto" if torch.cuda.is_available() else "cpu",
                                                          torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                                                          max_memory=None)
model = PeftModel.from_pretrained(model, peft_model_path,
                                  device_map="auto" if torch.cuda.is_available() else "cpu",
                                  max_memory=None)

print('Model Loaded successfully...')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

while True:
    print("-" * 150)
    text = input('Question: ')
    if text == 'exit':
        break
    response_start = input('Response Start: ')
    annotation_request = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": f'You are a helpful AI assistant representing the {party_name} political group in the European Parliament.'},
                                                                     {"role": "user",
                                                                      "content": text}],
                                                       tokenize=False, add_generation_prompt=False)

    annotation_request += response_start
    # Get the response from the chatbot
    responses = pipeline(
        annotation_request,
        do_sample=True,
        num_return_sequences=1,
        return_full_text=False,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    # Print the response
    print(f'Response:  {responses[0]["generated_text"].strip()}')
    print("-" * 50)