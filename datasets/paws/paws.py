# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import datalabs
from datalabs.tasks import TextMatching

_CITATION = '''\
@inproceedings{yang-etal-2019-paws,
    title = "PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
    author = "Yang, Yinfei  and
      Zhang, Yuan  and
      Tar, Chris  and
      Baldridge, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1382",
    doi = "10.18653/v1/D19-1382",
    pages = "3687--3692",
    abstract = "Most existing work on adversarial data generation focuses on English. For example, PAWS (Paraphrase Adversaries from Word Scrambling) consists of challenging English paraphrase identification pairs from Wikipedia and Quora. We remedy this gap with PAWS-X, a new dataset of 23,659 human translated PAWS evaluation pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. We provide baseline numbers for three models with different capacity to capture non-local context and sentence structure, and using different multilingual training and evaluation regimes. Multilingual BERT fine-tuned on PAWS English plus machine-translated data performs the best, with a range of 83.1-90.8 accuracy across the non-English languages and an average accuracy gain of 23{\%} over the next best model. PAWS-X shows the effectiveness of deep, multilingual pre-training while also leaving considerable headroom as a new challenge to drive multilingual research that better captures structure and contextual information.",
}


'''

_DESCRIPTION = '''\
This dataset, publicly available by Google for text matching task, 
contains more than 50,000 sentence pairs, some of which have similar semantics and some of which do not. 
For more information, please refer to https://aclanthology.org/D19-1382.
'''

_LICENSE = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/paws/License.pdf"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/paws/train.tsv"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/paws/dev.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/paws/test.tsv"


class PAWS(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1'])
            }),
            supervised_keys=None,
            homepage='https://aclanthology.org/D19-1382',
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TextMatching(
                text1_column="text1",
                text2_column="text2",
                task="natural-language-inference",
                label_column="label"),
            ],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = '\t')
            for id_, row in enumerate(csv_reader):
                if len(row) == 3:
                    text1, text2, label = row
                    label = int(label)
                    yield id_, {'text1': text1, 'text2': text2, 'label': label}
