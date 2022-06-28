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
The goal of this task is to extract all SPO triples that satisfy the schema constraints 
for a given natural language sentence according to a predefined set of schemas. 
The schema defines the relation (P) and its corresponding classes of subject (S) and object (O).
"""

_CITATION = """\
@InProceedings{10.1007/978-3-030-32236-6_72,
    author="Li, Shuangjie
        and He, Wei
        and Shi, Yabing
        and Jiang, Wenbin
        and Liang, Haijin
        and Jiang, Ye
        and Zhang, Yang
        and Lyu, Yajuan
        and Zhu, Yong",
    editor="Tang, Jie
        and Kan, Min-Yen
        and Zhao, Dongyan
        and Li, Sujian
        and Zan, Hongying ",
    title="DuIE: A Large-Scale Chinese Dataset for Information Extraction",
    booktitle="Natural Language Processing and Chinese Computing",
    year="2019",
    publisher="Springer International Publishing",
    address="Cham",
    pages="791--800",
    abstract="Information extraction is an important foundation for knowledge graph construction, as well as many natural language understanding applications. Similar to many other artificial intelligence tasks, high quality annotated datasets are essential to train a high-performance information extraction system. Existing datasets, however, are mostly built for English. To promote research in Chinese information extraction and evaluate the performance of related systems, we build a large-scale high-quality dataset, named DuIE, and make it publicly available. We design an efficient coarse-to-fine procedure including candidate generation and crowdsourcing annotation, in order to achieve high data quality at a large data scale. DuIE contains 210,000 sentences and 450,000 instances covering 49 types of commonly used relations, reflecting the real-world scenario. We also hosted an open competition based on DuIE, which attracted 1,896 participants. The competition results demonstrated the potential of this dataset in promoting information extraction research.",
    isbn="978-3-030-32236-6"
}
"""

_LICENSE = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuIE/License.docx"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuIE/duie_train.json"
_VALIDATION_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuIE/duie_dev.json"
_TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/DuIE/duie_sample.json"

_HOMEPAGE = "https://www.luge.ai/#/luge/dataDetail?id=5"


class DuIEConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(DuIEConfig, self).__init__(**kwargs)

class DuIE(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DuIEConfig(
            name="entity_relation_extraction",
            version=datalabs.Version("1.0.0"),
            description="entity_relation_extraction",
        ),
    ]

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "relation": datalabs.features.Sequence(
                        {
                            "predicate": datalabs.Value("string"),
                            "subject": datalabs.Value("string"),
                            "subject_type": datalabs.Value("string"),
                            "object": {
                                "@value": datalabs.Value("string"),
                                "inWork": datalabs.Value("string"),
                            },
                            "object_type": {
                                "@value": datalabs.Value("string"),
                                "inWork": datalabs.Value("string"),
                            },
                        })

                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.entity_relation_extraction)(
                    text_column = "text",
                    event_column = "relation",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        valid_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                line = json.loads(line)
                text, spo_list = line["text"], line["spo_list"]
                relation = []
                for spo in spo_list:
                    predicate = spo["predicate"]
                    subject = spo["subject"]
                    subject_type = spo["subject_type"] 
                    if "inWork" in spo["object"]:
                        object = {"@value": spo["object"]["@value"], "inWork" : spo["object"]["inWork"]}
                        object_type = {"@value": spo["object_type"]["@value"], "inWork" : spo["object_type"]["inWork"]}
                    else: 
                        object = {"@value": spo["object"]["@value"], "inWork" : '1'}
                        object_type = {"@value": spo["object_type"]["@value"], "inWork" : '1'}
                    relation.append({"predicate": predicate, "subject": subject, "subject_type": subject_type, "object": object, "object_type": object_type})
                yield id_, {"text": text, "relation":relation}