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

import csv

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
CCKS2019_fin is an event entity extraction dataset in the financial field, which is used for event extraction tasks. 
The task goal is to extract the event entity based on the given description text and event type.
"""

_CITATION = """\
@article{DBLP:journals/corr/abs-2003-03875,
  author    = {Xianpei Han and
               Zhichun Wang and
               Jiangtao Zhang and
               Qinghua Wen and
               Wenqi Li and
               Buzhou Tang and
               Qi Wang and
               Zhifan Feng and
               Yang Zhang and
               Yajuan Lu and
               Haitao Wang and
               Wenliang Chen and
               Hao Shao and
               Yubo Chen and
               Kang Liu and
               Jun Zhao and
               Taifeng Wang and
               Kezun Zhang and
               Meng Wang and
               Yinlin Jiang and
               Guilin Qi and
               Lei Zou and
               Sen Hu and
               Minhao Zhang and
               Yinnian Lin},
  title     = {Overview of the {CCKS} 2019 Knowledge Graph Evaluation Track: Entity, Relation, Event and {QA}},
  journal   = {CoRR},
  volume    = {abs/2003.03875},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.03875},
  eprinttype = {arXiv},
  eprint    = {2003.03875},
  timestamp = {Fri, 18 Sep 2020 10:42:43 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-03875.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2019_fin_event_entity/train.csv"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/ccks2019_fin_event_entity/test.csv"

_HOMEPAGE = "https://www.biendata.xyz/competition/ccks_2019_4"


class CCKS2019FinConfig(datalabs.BuilderConfig):
    def __init__(self, **kwargs):

        super(CCKS2019FinConfig, self).__init__(**kwargs)


class CCKS2019Fin(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCKS2019FinConfig(
            name="event_entity_extraction",
            version=datalabs.Version("1.0.0"),
            description="event_entity_extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "event_entity_extraction"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "event_type": datalabs.Value("string"),
                    "event_entity": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.event_entity_extraction)(
                    text_column = "text",
                    entity_column = "event_entity",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            csv_file = csv.reader(f, delimiter=",")
            for id_, line in enumerate(csv_file):
                text, event_type, event_entity = line[1], line[2], line[3]
                yield id_, {
                    "text": text,
                    "event_type": event_type,
                    "event_entity": event_entity,
                }
