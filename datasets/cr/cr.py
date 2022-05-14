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
CR is a customer review data set and each sample is labelled as positive or negative.
For more information, please refer to https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf
"""

_CITATION = """\
@inproceedings{10.1145/1341531.1341561,
author = {Ding, Xiaowen and Liu, Bing and Yu, Philip S.},
title = {A Holistic Lexicon-Based Approach to Opinion Mining},
year = {2008},
isbn = {9781595939272},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/1341531.1341561},
doi = {10.1145/1341531.1341561},
abstract = {One of the important types of information on the Web is the opinions expressed in the user generated content, e.g., customer reviews of products, forum posts, and blogs. In this paper, we focus on customer reviews of products. In particular, we study the problem of determining the semantic orientations (positive, negative or neutral) of opinions expressed on product features in reviews. This problem has many applications, e.g., opinion mining, summarization and search. Most existing techniques utilize a list of opinion (bearing) words (also called opinion lexicon) for the purpose. Opinion words are words that express desirable (e.g., great, amazing, etc.) or undesirable (e.g., bad, poor, etc) states. These approaches, however, all have some major shortcomings. In this paper, we propose a holistic lexicon-based approach to solving the problem by exploiting external evidences and linguistic conventions of natural language expressions. This approach allows the system to handle opinion words that are context dependent, which cause major difficulties for existing algorithms. It also deals with many special words, phrases and language constructs which have impacts on opinions based on their linguistic patterns. It also has an effective function for aggregating multiple conflicting opinion words in a sentence. A system, called Opinion Observer, based on the proposed technique has been implemented. Experimental results using a benchmark product review data set and some additional reviews show that the proposed technique is highly effective. It outperforms existing methods significantly},
booktitle = {Proceedings of the 2008 International Conference on Web Search and Data Mining},
pages = {231–240},
numpages = {10},
keywords = {opinion mining, context dependent opinions, sentiment analysis},
location = {Palo Alto, California, USA},
series = {WSDM '08}
}
"""

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1Y3zJdDzXwvciuTTUHUvVqmWF7xa1rt-_&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1Lpv-1MvlfHTiKJdB3L-ElQ3uF2htvHdx&export=download"


class CR(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["positive", "negative"]),
                }
            ),
            homepage="https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf",
            citation=_CITATION,
            languages=["en"],
            task_templates=[get_task(TaskType.sentiment_classification)(text_column="text", label_column="label")],
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
        """Generate CR examples."""

        # map the label into textual string
        textualize_label = {
            "1": "positive",
            "0": "negative"
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                text, label = row
                label = textualize_label[label]
                yield id_, {"text": text, "label": label}
