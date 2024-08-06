# Template for the annotation request
TEMPLATE_EU_FIRST_PERSON = "Would you agree or disagree with the statement \"{}\"?\n" \
              "Options:\n(a) I {} \"{}\".\n" \
              "(b) I {} \"{}\".\n" \
              "(c) I {} \"{}\".\n" \
              "(d) I {} \"{}\".\n" \
              "(e) I {} \"{}\".\n"

TEMPLATE_EU_ROLE_PERSON = "As a member of the {}, would you agree or disagree with the statement \"{}\"?\n" \
              "Options:\n(a) I {} \"{}\".\n" \
              "(b) I {} \"{}\".\n" \
              "(c) I {} \"{}\".\n" \
              "(d) I {} \"{}\".\n" \
              "(e) I {} \"{}\".\n"

TEMPLATE_EU_THIRD_PERSON = "Would the party {} agree or disagree with the statement \"{}\"?\n" \
              "Options:\n(a) The {} {} \"{}\".\n" \
              "(b) The {} {} \"{}\".\n" \
              "(c) The {} {} \"{}\".\n" \
              "(d) The {} {} \"{}\".\n" \
              "(e) The {} {} \"{}\".\n"

A_EU = "would completely disagree with the statement"
B_EU = "would tend to disagree with the statement"
C_EU = "would be neutral towards the statement"
D_EU = "would tend to agree with the statement"
E_EU = "would completely agree with the statement"

A_EU_T = "would completely disagree with the statement"
B_EU_T = "would tend to disagree with the statement"
C_EU_T = "would be neutral towards the statement"
D_EU_T = "would tend to agree with the statement"
E_EU_T = "would completely agree with the statement"

TEMPLATE_EU_ANSWERS = [A_EU, B_EU, C_EU, D_EU, E_EU]
TEMPLATE_EU_ANSWER_T = [A_EU_T, B_EU_T, C_EU_T, D_EU_T, E_EU_T]


def build_prompt_first_person(example):
    example["annotation_request"] = TEMPLATE_EU_FIRST_PERSON.format(
        example["statement"], TEMPLATE_EU_ANSWERS[0], example["statement"], TEMPLATE_EU_ANSWERS[1],
        example["statement"], TEMPLATE_EU_ANSWERS[2], example["statement"], TEMPLATE_EU_ANSWERS[3],
        example["statement"], TEMPLATE_EU_ANSWERS[4], example["statement"])
    return example

def build_prompt_role_person(example):
    example["annotation_request"] = TEMPLATE_EU_ROLE_PERSON.format( example['party_name'],
        example["statement"], TEMPLATE_EU_ANSWERS[0], example["statement"], TEMPLATE_EU_ANSWERS[1],
        example["statement"], TEMPLATE_EU_ANSWERS[2], example["statement"], TEMPLATE_EU_ANSWERS[3],
        example["statement"], TEMPLATE_EU_ANSWERS[4], example["statement"])
    return example


def build_prompt_third_person(example):
    example["annotation_request"] = TEMPLATE_EU_THIRD_PERSON.format(
        example['party_name'], example["statement"],
        example['party_name'], TEMPLATE_EU_ANSWER_T[0], example["statement"],
        example['party_name'], TEMPLATE_EU_ANSWER_T[1], example["statement"],
        example['party_name'], TEMPLATE_EU_ANSWER_T[2], example["statement"],
        example['party_name'], TEMPLATE_EU_ANSWER_T[3], example["statement"],
        example['party_name'], TEMPLATE_EU_ANSWER_T[4], example["statement"])
    return example
