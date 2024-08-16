from data import DATA_DIR
import os
import json
import numpy as np
from datasets import load_dataset

dimensions = ['Liberal society', 'Environmental protection', 'EU integration', 'Economic liberalization',
              'Finance restrictions', 'Immigration restrictions', 'Law and Order', 'Left/Right', 'Anti-EU/Pro-EU']
dimensions_abbr = ['LIB SOC', 'ENV PROT', 'EU INT', 'ECON LIB', 'FIN RESTR', 'IMM RESTR', 'LAW ORDER', 'LEFT/RIGHT', 'ANTI-EU/PRO-EU']

TYPE = 'role'
f_out = open(os.path.join(DATA_DIR, f'llms_allignment_scores_{TYPE}.json'), 'w')
# MODELS = ['Meta-Llama-3.1-8B-Instruct_LEFT_first_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_S&D_first_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_EPP_first_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_ID_first_chronos_responses.jsonl']
#
# MODELS = ['Meta-Llama-3.1-8B-Instruct_LEFT_third_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_S&D_third_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_EPP_third_chronos_responses.jsonl',
#           'Meta-Llama-3.1-8B-Instruct_ID_third_chronos_responses.jsonl']

MODELS = ['Meta-Llama-3.1-8B-Instruct_LEFT_role_chronos_responses.jsonl',
          'Meta-Llama-3.1-8B-Instruct_S&D_role_chronos_responses.jsonl',
          'Meta-Llama-3.1-8B-Instruct_EPP_role_chronos_responses.jsonl',
          'Meta-Llama-3.1-8B-Instruct_ID_role_chronos_responses.jsonl']

TERMS = ['7th (prompt)', '8th (prompt)', '9th (prompt)']

USE_CUSTOM_DIMENSIONS = False

party_data_2024 = load_dataset('coastalcph/euandi_2024', 'party_positions')['test']

party_responses_2024 = {}
for party in party_data_2024:
    party_responses_2024[party['party_name']] = party

# Load CSV dataset
party_data_2019 = load_dataset("csv", data_files=os.path.join(DATA_DIR, 'euandi_2019', 'aggregated_party_responses.csv'))['train']
party_responses_2019 = {}
for party in party_data_2019:
    party_responses_2019[party['party_name']] = party

# Load EUANDI questionnaire dataset
euandi_questionnaire = {}
with open(os.path.join(DATA_DIR, 'euandi_2024', 'euandi_2024_questionnaire_aligned.jsonl'), 'r') as f_q:
    for line in f_q:
        statement = json.loads(line)
        euandi_questionnaire[statement['statement']] = statement


for model_path in MODELS:
    if 'S&D' in model_path:
        party_name_2019 = 'S&D'
        party_name_2024 = 'PES'
    elif 'EPP' in model_path:
        party_name_2019 = 'EPP'
        party_name_2024 = 'EPP'
    elif 'ID' in model_path:
        party_name_2019 = 'ID'
        party_name_2024 = 'ID'
    elif 'LEFT' in model_path:
        party_name_2019 = 'GUE/NGL'
        party_name_2024 = 'PEL'
    model_results = {term: {key: [] for key in dimensions} for term in TERMS}
    with open(os.path.join(DATA_DIR, 'model_responses', model_path), 'r') as f_model:
        for line in f_model:
            statement_data = json.loads(line)
            if euandi_questionnaire[statement_data['statement']]['old_statement_idx'] is None and USE_CUSTOM_DIMENSIONS is False:
                continue
            for term in ['7th', '8th', '9th']:
                term_agg = []
                for prompt in range(3):
                    term_agg.append(statement_data[f'normalized_response_{term}_{prompt}'])
                rep_signs = [val for val in term_agg if val > 0]
                # if len(rep_signs) >= 2:
                #     term_agg = rep_signs
                # else:
                #     term_agg = [val for val in term_agg if val < 0]
                for dimension in dimensions:
                    if dimension in euandi_questionnaire[statement_data['statement']]:
                        if euandi_questionnaire[statement_data['statement']][dimension] != 0:
                            model_results[f'{term} (prompt)'][dimension].append(euandi_questionnaire[statement_data['statement']][dimension] * np.average(term_agg))
    party_data = party_responses_2024[party_name_2024]
    model_results['9th (gold)'] = {key: [] for key in dimensions}
    for statement, statement_data in euandi_questionnaire.items():
        for dimension in dimensions:
            if dimension in statement_data:
                if statement_data[dimension] != 0 and party_data[f'statement_{statement_data["statement_idx"]}']['answer'] != 'No opinion':
                    if party_data[f'statement_{statement_data["statement_idx"]}']['answer'] == 'Completely disagree':
                        answer = -1
                    elif party_data[f'statement_{statement_data["statement_idx"]}']['answer'] == 'Tend to disagree':
                        answer = -0.5
                    elif party_data[f'statement_{statement_data["statement_idx"]}']['answer'] == 'Tend to agree':
                        answer = 0.5
                    elif party_data[f'statement_{statement_data["statement_idx"]}']['answer'] == 'Completely agree':
                        answer = 1
                    else:
                        continue
                    model_results['9th (gold)'][dimension].append(statement_data[dimension] * answer)
    party_data = party_responses_2019[party_name_2019]
    model_results['8th (gold)'] = {key: [] for key in dimensions}
    for statement, statement_data in euandi_questionnaire.items():
        if statement_data["old_statement_idx"] is not None:
            for dimension in dimensions:
                if dimension in statement_data:
                    if statement_data[dimension] != 0:
                        model_results['8th (gold)'][dimension].append(statement_data[dimension] * party_data[f'statement_{statement_data["old_statement_idx"]}'])

    for key in model_results.keys():
        temp_dict = {'name': f'{party_name_2019} / {key} / ({TYPE})'}
        temp_dict.update(model_results[key])
        f_out.write(json.dumps(temp_dict) + '\n')
    print('=' * 50)
    print('PARTY NAME:', party_name_2019)
    print('=' * 50)
    dim_header = f'{"DIMENSION":>15}'
    print(dim_header + ' |\t 7th (prompt) |\t 8th (prompt) |\t   8th (gold) |\t 9th (prompt) |\t   9th (gold) |')
    print('-' * 100)
    for dim_idx, dimension in enumerate(dimensions):
        print(f'{dimensions_abbr[dim_idx]:>15} |\t', end='')
        for term in ['7th (prompt)', '8th (prompt)', '8th (gold)', '9th (prompt)', '9th (gold)']:
            value = f'{np.average(model_results[term][dimension]):.1f}'
            print(f'{value:>13} |', end='\t')
        print()

f_out.close()