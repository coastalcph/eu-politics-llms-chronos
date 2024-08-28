"""EU Debates"""

import json
import os
import textwrap
from data import DATA_DIR
import datasets


MAIN_CITATION = """
@inproceedings{chalkidis-and-brandl-eu-llama-2024,
    title = "Llama meets EU: Investigating the European political spectrum through the lens of LLMs",
    author = "Chalkidis, Ilias  and
      Stephanie Brandl",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}
"""

_DESCRIPTION = """
EU Debates is a corpus of parliamentary proceedings (debates) from the EU parliament. The corpus consists of approx. 87k individual speeches in the period 2009-2023. 
We exhaustively scrape the data from the official European Parliament Plenary website. All speeches are time-stamped, thematically organized on debates, 
and include metadata relevant to the speaker's identity (full name, euro-party affiliation, speaker role), and the debate (date and title). 
Older debate speeches are originally in English, while newer ones are linguistically diverse across the 23 official EU languages, thus we also provide machine-translated 
versions in English, when official translations are missing. 
"""


class EUDebatesConfig(datasets.BuilderConfig):
    """BuilderConfig for EU Debates"""

    def __init__(
        self,
        data_url,
        citation,
        **kwargs,
    ):
        """BuilderConfig for EU Debates.

        Args:
          data_url: `string`, url to download the zip file from
          data_file: `string`, filename for data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(EUDebatesConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.data_url = data_url
        self.citation = citation


class EUDebates(datasets.GeneratorBasedBuilder):
    """EU Debates. Version 1.0"""

    BUILDER_CONFIGS = [
        EUDebatesConfig(
            name="v1",
            data_url=os.path.join(DATA_DIR, "eu_debates_extended", "eu_parliaments_extended.json"),
            citation=textwrap.dedent(MAIN_CITATION),
        ),
        EUDebatesConfig(
            name="v2",
            data_url=os.path.join(DATA_DIR, "eu_debates_extended", "eu_parliaments_extended_v2.json"),
            citation=textwrap.dedent(MAIN_CITATION),
        ),
        EUDebatesConfig(
            name="v3",
            data_url=os.path.join(DATA_DIR, "eu_debates_extended", "eu_parliaments_extended_v3.jsonl"),
            citation=textwrap.dedent(MAIN_CITATION),
        ),
    ]

    def _info(self):
        features = {"text": datasets.Value("string"),
                    "translated_text": datasets.Value("string"),
                    "rewritten_text": datasets.Value("string"),
                    "speaker_party": datasets.Value("string"),
                    "speaker_role": datasets.Value("string"),
                    "speaker_name": datasets.Value("string"),
                    "debate_title": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "year": datasets.Value("string")}

        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
            homepage='https://www.europarl.europa.eu/',
            citation=MAIN_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": self.config.data_url,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data['speaker_role'] == 'MEP':
                    if 'question' in data and data['question'] is not None:
                        example = {
                            "text": data["text"] if 'text' in data else None,
                            "translated_text": data["translated_text"] if 'translated_text' in data else None,
                            "rewritten_text": data["rewritten_text"] if 'rewritten_text' in data else None,
                            "speaker_party": data["speaker_party"],
                            "speaker_role": data["speaker_role"],
                            "speaker_name": data["speaker_name"],
                            "debate_title": data["debate_title"],
                            "question": data["question"],
                            "date": data["date"],
                            "year": data["year"]
                        }
                        yield id_, example