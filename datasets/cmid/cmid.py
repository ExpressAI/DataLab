
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
This dataset is used for Chinese medical QA intent understanding task.
It is provided by machine learning group held by Su Xiangdong from Inner Mongolia University. 
For more information, please refer to https://tianchi.aliyun.com/dataset/dataDetail?dataId=92109. 
"""

_CITATION = """\
@article{Chen2020ABD,
  title={A benchmark dataset and case study for Chinese medical question intent classification},
  author={Nan Chen and Xiangdong Su and Tongyang Liu and Qizhi Hao and Ming Wei},
  journal={BMC Medical Informatics and Decision Making},
  year={2020},
  volume={20}
}
"""


_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/CMID/CMID.json"

LABELS_36class = [
    '定义',
    '病因',
    '临床表现',
    '相关病症',
    '治疗方法',
    '推荐医院',
    '预防',
    '所属科室',
    '禁忌',
    '传染性',
    '治愈率',
    '严重性',
    '作用',
    '适用症',
    '价钱',
    '药物禁忌',
    '用法',
    '副作用',
    '成分',
    '方法',
    '费用',
    '有效时间',
    '临床意义/检查目的',
    '治疗时间',
    '疗效',
    '恢复时间',
    '正常指标',
    '化验/体检方案',
    '恢复',
    '设备用法',
    '多问',
    '养生',
    '整容',
    '两性',
    '对比',
    '无法确定'
]
LABELS_4class = [
                    "病症",
                    "药物",
                    "治疗方案",
                    "其他"
                ]


class CMIDConfig(datalabs.BuilderConfig):
    """BuilderConfig for CMID."""

    def __init__(self, **kwargs):
        """BuilderConfig for CMID.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CMIDConfig, self).__init__(**kwargs)

class CMID(datalabs.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        CMIDConfig(
            name="label_4class",
            # version=datalabs.Version("1.0.0"),
            description="label_4class is the primary type",
        ),
        CMIDConfig(
            name="label_36class",
            # version=datalabs.Version("1.0.0"),
            description="label_36class is the secondary type",
        ),
    ]
    DEFAULT_CONFIG_NAME = "label_4class"


    def _info(self):
        if self.config.name == 'label_4class':
            _FEATURES = {
                "text": datalabs.Value("string"),
                "entities": datalabs.features.Sequence(datalabs.Value("string")),
                "seg_result": datalabs.features.Sequence(datalabs.Value("string")),
                "label": datalabs.features.ClassLabel(names=LABELS_4class)
            }
        else:
            _FEATURES = {
                "text": datalabs.Value("string"),
                "entities": datalabs.features.Sequence(datalabs.Value("string")),
                "seg_result": datalabs.features.Sequence(datalabs.Value("string")),
                "label": datalabs.features.ClassLabel(names=LABELS_36class),
            }
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(_FEATURES),
            homepage="https://github.com/CLUEbenchmark/FewCLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.intent_classification)(
                text_column="text",
                label_column="label")],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path})
        ]
        


    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            file = json.load(f)
            for id_ in range(len(file)):
                text = file[id_]['originalText']
                entities = file[id_]['entities']
                seg_result = file[id_]['seg_result']
                if self.config.name == "label_36class":
                    label = file[id_]['label_36class'][0].replace("\"","").replace("'","")
                else:
                    label = file[id_]['label_4class'][0].replace("\"","").replace("'","")
                if label not in LABELS_36class + LABELS_4class:
                    continue

                yield id_, {'text': text, 'label': label, 'entities': entities, 'seg_result': seg_result}