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

_DESCRIPTION = """\
EPRSTMT(EPR-sentiment): E-commerce Product Review Dataset for Sentiment Analysis
Data scale: train set(32), validation set(32), test set with labels(610), test set without labels(753), unlabeled corpus(19565). 
For more information, please refer to https://github.com/CLUEbenchmark/FewCLUE. 
"""

_CITATION = """\
@article{Xu2021FewCLUEAC,
  title={FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark},
  author={Liang Xu and Xiaojing Lu and Chenyang Yuan and Xuanwei Zhang and Huining Yuan and Huilin Xu and Guoao Wei and Xiang Pan and Hai Hu},
  journal={ArXiv},
  year={2021},
  volume={abs/2107.07498}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/eprstmt/train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/eprstmt/dev.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/eprstmt/test.json"
# _TEST_UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/eprstmt/test_unlabeled.json"
# _UNLABELED_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/eprstmt/unlabeled.json"

class EPRSTMT(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["Positive","Negative"])
                }
            ),
            homepage="https://github.com/CLUEbenchmark/FewCLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.sentiment_classification)(
                text_column="text",
                label_column="label")],
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
       
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                text, label = line['sentence'], line['label']
                if label == ("Positive" or "Negative"):
                    yield id_, {'text': text, 'label': label}
