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
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Restaurant14 contains annotated reviews of restaurants reviews. Each sample is labeled as positive,
neutral or negative w.r.t. a specific aspect. For more information, please refer to https://aclanthology.org/S14-2004.pdf
"""

_CITATION = """\
@inproceedings{pontiki-etal-2014-semeval,
    title = "{S}em{E}val-2014 Task 4: Aspect Based Sentiment Analysis",
    author = "Pontiki, Maria  and
      Galanis, Dimitris  and
      Pavlopoulos, John  and
      Papageorgiou, Harris  and
      Androutsopoulos, Ion  and
      Manandhar, Suresh",
    booktitle = "Proceedings of the 8th International Workshop on Semantic Evaluation ({S}em{E}val 2014)",
    month = aug,
    year = "2014",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S14-2004",
    doi = "10.3115/v1/S14-2004",
    pages = "27--35",
}
"""

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1gxLzaBhoQjSG1wJsj5bCMIUdsEyc46d8&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1UMp2LpAIMAQQgO2guTukIU0itFIcISIz&export=download"


class Restaurant14(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "aspect": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative", "neutral"]),
                }
            ),
            homepage="https://aclanthology.org/S14-2004.pdf",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.aspect_based_sentiment_classification)(
                    span_column="aspect",
                    text_column="text",
                    label_column="label"
                )]
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate Restaurant14 examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                aspect, text, label = row
                yield id_, {"aspect": aspect, "text": text, "label": label}
