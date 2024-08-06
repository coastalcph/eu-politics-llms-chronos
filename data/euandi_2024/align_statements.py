from datasets import load_dataset
from data import DATA_DIR
import os
import json

statements_2019 = load_dataset('euandi_2019', 'questionnaire')['test']
statements_2024 = []
with open(os.path.join(DATA_DIR, 'euandi_2024', 'euandi_2024_questionnaire.jsonl'), 'r') as f_q:
    for line in f_q:
        statement = json.loads(line)
        statements_2024.append(statement)

statements_2024_aligned = []
for statement_1 in statements_2024:
    for idx, statement_2 in enumerate(statements_2019):
        if statement_1['statement'].lower() == statement_2['statement']['en'][:-1].lower():
            statement_1.update({'old_statement_idx': idx+1})
            for key in statement_2:
                if key != 'statement':
                    statement_1.update({key: statement_2[key]})

        elif statement_1['statement'].replace('EU', 'European Union').lower() == statement_2['statement']['en'][:-1].lower():
            statement_1.update({'old_statement_idx': idx+1})
            for key in statement_2:
                if key != 'statement':
                    statement_1.update({key: statement_2[key]})
    statements_2024_aligned.append(statement_1)

with open(os.path.join(DATA_DIR, 'euandi_2024', 'euandi_2024_questionnaire_aligned.jsonl'), 'w') as f_o:
    for statement in statements_2024_aligned:
        f_o.write(json.dumps(statement) + '\n')

