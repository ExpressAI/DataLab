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
from datalabs.features.features import Sequence

_DESCRIPTION = """\
CCKS2021_fin is an event element extraction dataset in the financial field, which is used for event extraction tasks. 
The task goal is to extract some of the 13 elements of an event based on the given description text and text type.
"""

_CITATION = """\
For more information, please refer to http://sigkg.cn/ccks2021/?page_id=13.
"""

_LICENSE = "NA"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/CCKS2021_fin/train.txt"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/event_extraction/CCKS2021_fin/test_without_tags.txt"

_HOMEPAGE = "https://www.biendata.xyz/competition/ccks_2021_task6_1"


class CCKS2021FinConfig(datalabs.BuilderConfig):
    
    def __init__(self, **kwargs):

        super(CCKS2021FinConfig, self).__init__(**kwargs)

class CCKS2021Fin(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CCKS2021FinConfig(
            name="Event Extraction",
            version=datalabs.Version("1.0.0"),
            description="Event Extraction",
        ),
    ]

    DEFAULT_CONFIG_NAME = "Event Extraction"

    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "source": 
                        {
                            "text": datalabs.Value("string"),
                            "level1": datalabs.Value("string"),
                            "level2": datalabs.Value("string"),
                            "level3": datalabs.Value("string"),
                        },
                    "reference": datalabs.features.Sequence(
                        {
                            "start": datalabs.Value("int32"), 
                            "end": datalabs.Value("int32"),  
                            "type": datalabs.Value("string"),
                            "entity": datalabs.Value("string"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.event_extraction)(
                    source_column = "source",
                    reference_column = "reference",
                ),
            ],
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as txt_file:
            for id_, line in enumerate(txt_file):
                line = json.loads(line)
                text, level1, level2, level3, attributes = line["text"], line["level1"], line["level2"], line["level3"], line["attributes"]
                reference = []
                for attribute in attributes:
                    start, end, type, entity = attribute["start"], attribute["end"], attribute["type"], attribute["entity"]
                    attribute = {"start": start, "end": end, "type": type, "entity": entity}
                    reference.append(attribute)
                source = {"text":text, "level1":level1, "level1":level2, "level1":level3}
                if len(reference) > 0:
                    yield id_, {'source': source, 'reference': reference}