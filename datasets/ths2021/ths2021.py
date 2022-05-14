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
from datalabs import get_task, TaskType

_DESCRIPTION = """\
This data set contains sentence-pairs from customer service conversations from multiple fields, which can be used for text matching tasks.
For more information, please refer to http://contest.aicubes.cn/#/detail?topicId=23. 
"""

_CITATION = """\
For more information, please refer to http://contest.aicubes.cn/#/detail?topicId=23. 
"""

_LICENSE = 'NA'

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ths2021/train.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ths2021/test.tsv"


class THS2021(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1'])
            }),
            supervised_keys=None,
            homepage='http://contest.aicubes.cn/#/detail?topicId=23',
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.paraphrase_identification)(
                text1_column="text1",
                text2_column="text2",
                label_column="label"),
            ],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = '\t')
            for id_, row in enumerate(csv_reader):
                if len(row) == 3:
                    label, text1, text2 = row
                    label = int(label)
                    yield id_, {'text1': text1, 'text2': text2, 'label': label}
