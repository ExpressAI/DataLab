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

import json
import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
A large-scale Chinese Nature language inference and Semantic similarity calculation Dataset.
For more information, please refer to https://github.com/pluto-junzeng/CNSD. 
"""

_CITATION = """\
For more information, please refer to https://6a75-junzeng-uxxxm-1300734931.tcb.qcloud.la/CNSD.pdf?sign=401485f4d6f256393a264e68464ca4ae&t=1578114336.
"""

_LICENSE = "NA" 

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cnsd/cnsd-sts-train.txt"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cnsd/cnsd-sts-dev.txt"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/cnsd/cnsd-sts-test.txt"


class CNSD(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1','2','3','4','5']),
            }),
            supervised_keys=None,
            homepage='https://github.com/pluto-junzeng/CNSD',
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
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        
        with open(filepath, encoding="utf-8") as txt_file:
            for id_, line in enumerate(txt_file):
                line_l = line.split("||")
                if len(line_l) == 4:
                    text1 = line_l[1]
                    text2 = line_l[2]
                    label = line_l[3]
                    yield id_, {'text1': text1, 'text2': text2, 'label': label}
                
