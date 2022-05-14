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
from datalabs.tasks import TextMatching

_CITATION = '''\
@article{xu2020clue,
  title={CLUE: A Chinese language understanding evaluation benchmark},
  author={Xu, Liang and Hu, Hai and Zhang, Xuanwei and Li, Lu and Cao, Chenjie and Li, Yudong and Xu, Yechen and Sun, Kai and Yu, Dian and Yu, Cong and others},
  journal={arXiv preprint arXiv:2004.05986},
  year={2020}
}
'''

_DESCRIPTION = '''\
This dataset is OCNLI (Original Chinese Natural Language Inference), the first untranslated large Chinese natural language inference dataset. 
OCNLI has roughly 50k pairs for training, 3k for development and 3k for test. 
For more information, please refer to https://github.com/CLUEbenchmark/OCNLI. 
'''

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ocnli/train.json"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ocnli/dev.json"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/ocnli/test.json"


class OCNLI(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1', '-1']),
            }),
            supervised_keys=None,
            homepage='https://github.com/CLUEbenchmark/OCNLI',
            citation=_CITATION,
            languages=["zh"],
            task_templates=[TextMatching(
                text1_column="text1",
                text2_column="text2",
                label_column="label",
                task="natural-language-inference"),
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
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text1, text2, label = line['sentence1'], line['sentence2'], line['label']
                if label == 'entailment':
                    label = 1
                elif label == 'neutral':
                    label = 0
                elif label == 'contradiction':
                    label = -1
                if label == (1 or 0 or -1):
                    yield id_, {'text1': text1, 'text2': text2, 'label': label}

