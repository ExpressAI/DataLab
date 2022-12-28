# coding=utf-8
# Copyright 2020 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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
from typing import ClassVar, Optional, Tuple
from datalabs.features import Features, Sequence, Value, ClassLabel
logger = datalabs.logging.get_logger(__name__)

_DESCRIPTION = """\
The CMRC2018 dataset is a single-chapter, extractive reading comprehension dataset.
Given a question (q), a chapter (p) and its title (t), the system needs to give the answers (a) of the question. 
For more information, please refer to https://github.com/CLUEbenchmark/CLUE. 
"""

_CITATION = """\
@article{xu2020cluener2020,
  title={CLUENER2020: Fine-grained Name Entity Recognition for Chinese},
  author={Xu, Liang and Dong, Qianqian and Yu, Cong and Tian, Yin and Liu, Weitang and Li, Lu and Zhang, Xuanwei},
  journal={arXiv preprint arXiv:2001.04351},
  year={2020}
 }
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/ner/cluener/train.json"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/ner/cluener/dev.json"


_HOMEPAGE = "https://github.com/CLUEbenchmark/CLUENER2020"

class CMRC2018(datalabs.GeneratorBasedBuilder):


    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=Features(
                {
                    "text": Value("string"),
                    "subject":  datalabs.features.ClassLabel(
                                    names=[
                                        "address","company","game","government","movie","name",
                                        "organization","position","scene","book"
                                        ]
                                    ),

                    "spans": Sequence({
                                        "start_idx": Sequence(Value("int32")),
                                        "end_idx": Sequence(Value("int32")),
                                        "name": Value("string")
                            }),

                }
                    
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages = ["zh"],
            task_templates=[
                get_task(TaskType.ner_span_prediction)(
                    text_column = "text",
                    label_column = "labels",
                )
            ],
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
        ]

    def _generate_examples(self, filepath):
      
        count = 0
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
               
                text = item["text"]

                for subject in item['label']:
                   
                    contents=item['label'][subject]
                   
                    names,start_ids,end_ids=[],[],[]

                    for name in contents:
                        names.append(name)
                        
                        start_ids.append([i[0] for i in contents[name]])
                        end_ids.append([i[1] for i in contents[name]])
                
                    yield count, {
                            "text": text, 
                            "subject": subject,
                            "spans":{
                                "start_idx": start_ids,
                                "end_idx": end_ids,
                                "name":names
                            }
                        }

                    count = count + 1