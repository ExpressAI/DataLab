# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv

import datalabs
from datalabs import Dataset
from datalabs.tasks import TextClassification

_DESCRIPTION = """\
The ATIS (Airline Travel Information Systems) is a dataset consisting of audio 
recordings and corresponding manual transcripts about humans asking for flight 
information on automated airline travel inquiry systems. The data consists of 
17 unique intent categories. The original split contains 4478, 500 and 893 
intent-labeled reference utterances in train, development and test set respectively.
For more information, please refer to the link https://aclanthology.org/H90-1021/
"""

_CITATION = """\
@inproceedings{hemphill-etal-1990-atis,
    title = "The {ATIS} Spoken Language Systems Pilot Corpus",
    author = "Hemphill, Charles T.  and
      Godfrey, John J.  and
      Doddington, George R.",
    booktitle = "Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, {P}ennsylvania, June 24-27,1990",
    year = "1990",
    url = "https://aclanthology.org/H90-1021",
}
"""

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=11L91ee17GlgwVs-tgnN09W8-Nm1C7Gx5&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=184vRZwd0EnFBGBc-A41NtnBgh1Z6G3FL&export=download"


class ATIS(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "abbreviation",
                            "aircraft",
                            "aircraft, flight and flight number",
                            "airfare",
                            "airfare and flight",
                            "airfare and flight time",
                            "airline",
                            "airline and flight number",
                            "airport",
                            "capacity",
                            "cheapest",
                            "city",
                            "distance",
                            "day name",
                            "flight",
                            "flight and airfare",
                            "flight and airline",
                            "flight number",
                            "flight time",
                            "flight number and airline",
                            "ground fare",
                            "ground service",
                            "ground service and ground fare",
                            "meal",
                            "quantity",
                            "restriction",
                        ]
                    ),
                }
            ),
            homepage="https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                TextClassification(text_column="text", label_column="label")
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate ATIS examples."""

        # map the label into textual string
        textualize_label = {
            "abbreviation": "abbreviation",
            "aircraft": "aircraft",
            "aircraft+flight+flight_no": "aircraft, flight and flight number",
            "airfare": "airfare",
            "airfare+flight": "airfare and flight",
            "airfare+flight_time": "airfare and flight time",
            "airline": "airline",
            "airline+flight_no": "airline and flight number",
            "airport": "airport",
            "capacity": "capacity",
            "cheapest": "cheapest",
            "city": "city",
            "distance": "distance",
            "day_name": "day name",
            "flight": "flight",
            "flight+airfare": "flight and airfare",
            "flight+airline": "flight and airline",
            "flight_no": "flight number",
            "flight_time": "flight time",
            "flight_no+airline": "flight number and airline",
            "ground_fare": "ground fare",
            "ground_service": "ground service",
            "ground_service+ground_fare": "ground service and ground fare",
            "meal": "meal",
            "quantity": "quantity",
            "restriction": "restriction",
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            # using this for tsv: csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                text, label = row
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
