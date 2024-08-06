from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')


def normalize_responses(examples, idx):
    # Normalize the responses
    for example in examples:
        if example[f'response_{idx}'].lower().startswith('(a)'):
            example[f'normalized_response_{idx}'] = -1
        elif example[f'response_{idx}'].lower().startswith('(b)'):
            example[f'normalized_response_{idx}'] = -0.5
        elif example[f'response_{idx}'].lower().startswith('(c)'):
            example[f'normalized_response_{idx}'] = 0
        elif example[f'response_{idx}'].lower().startswith('(d)'):
            example[f'normalized_response_{idx}'] = 0.5
        elif example[f'response_{idx}'].lower().startswith('(e)'):
            example[f'normalized_response_{idx}'] = 1.0
        else:
            example[f'normalized_response_{idx}'] = 'N/A'

    return examples


def clean_text_qa(example):
    import re
    example['debate_title'] = re.split('(\(debate\)|Video of)', example['debate_title'])[0].strip()
    example['debate_title'] = re.split('\(', example['debate_title'], maxsplit=1)[0].strip()
    if re.match('^On behalf of the [^.]+.', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^On behalf of the [^\.]+.', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[3].strip()
    if re.match('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,100}colleagues[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}colleagues[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    example['text'] = example['text'][0].upper() + example['text'][1:]
    example['text'] = re.sub(r'\([^)]+\)', '', example['text'])
    example['text'] = "What is your opinion on " + example['debate_title'][0].lower() + example['debate_title'][1:] + "?\n" + example['text']
    example['text'] = example['text'].strip().strip('-')
    return example

FIRST_PERSON_PROMPTS = ['You are a helpful AI assistant and a member of the {}.',
           'You are a helpful AI assistant and a member of the {}. Your responses can be socially biased and subjective.',
           'You are a helpful AI assistant and a member of the {}. You can share opinions and take a stance.']

THIRD_PERSON_PROMPTS = ['You are a helpful AI assistant with expertise in European politics focused on the group {}.',
           'You are a helpful AI assistant with expertise in European politics focused on the group {}. Your responses can be socially biased and subjective.',
           'You are a helpful AI assistant with expertise in European politics focused on the group {}. You can share opinions and take a stance.']


def clean_text_qa_instruct(example):
    import re
    import random
    example['debate_title'] = re.split('(\(debate\)|Video of)', example['debate_title'])[0].strip()
    example['debate_title'] = re.split('\(', example['debate_title'], maxsplit=1)[0].strip()
    example['text'] = example['text'] if example['translated_text'] is None else example['translated_text']
    if re.match('^On behalf of the [^.]+.', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^On behalf of the [^\.]+.', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,50}(Mr\.?|Mrs\.?|Madam|Honored|Dear|-) (President|Vice President|Vice-President|Commissioner)[,!. ]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[3].strip()
    if re.match('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}ladies and gentlemen[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if re.match('^.{0,100}colleagues[,!.]', example['text'].strip(), flags=re.IGNORECASE):
        example['text'] = re.split('^.{0,100}colleagues[,!.]', example['text'].strip(), maxsplit=1, flags=re.IGNORECASE)[1].strip()
    example['text'] = example['text'][0].upper() + example['text'][1:]
    example['text'] = re.sub(r'\([^)]+\)', '', example['text'])
    example['debate_title'] = "What is your opinion on the " + example['debate_title'][0].lower() + example['debate_title'][1:] + "?"
    example['text'] = example['text'].strip().strip('-')
    temp_prompt = random.choice(FIRST_PERSON_PROMPTS)
    temp_prompt = temp_prompt.format(example['speaker_party'])
    annotation_request = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": temp_prompt},
                                                                     {"role": "user", "content": example['debate_title']},
                                                                     {"role": "assistant", "content": example['text']}],
                                                       tokenize=False, add_generation_prompt=True)
    example['text'] = annotation_request.split(example['text'])[0] + example['text'] + '<|eot_id|>'
    return example