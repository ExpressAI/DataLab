# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json

import datalabs
from datalabs.features import logger
from datalabs.utils import private_utils
from datalabs import get_task, TaskType

_DESCRIPTION = """\
The task of performance prediction is to be able to accurately predict the performance
of a model on a set of target languages. These languages may be present in the
fine-tuning data (few-shot training) or may not be present (zero-shot training). The
languages used for fine-tuning are referred to as pivots, while the languages that we
would like to evaluate model on are targets. The SumEval 2022 worksho shared task
consists of building a machine learning model that can accurately predict the
performance of a multilingual model on languages and tasks that we do not have test data
for, given accuracies of models on various combinations of pivot and target pairs.
https://www.microsoft.com/en-us/research/event/sumeval-2022/shared-task/
"""

_CITATION = """\
TODO
"""

_TEST_DOWNLOAD_URL = f"{private_utils.PRIVATE_LOC}/sumeval_2022/sumeval_test.json"


class SumEval2022(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {"overall_setting": datalabs.Value("string"),
                 "dataset_name": datalabs.Value("string"),
                 "model_name": datalabs.Value("string"),
                 "sub_setting": datalabs.Value("string"),
                 "data_setting": datalabs.Value("string"),
                 "target_lang_data_size": datalabs.Value("float"),
                 "target_lang": datalabs.Value("string"),
                 "true_value": datalabs.Value("float")}
            ),
            homepage="https://www.microsoft.com/en-us/research/event/sumeval-2022/shared-task/",
            citation=_CITATION,
            languages=["af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", "pt", "ro", "ru", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh"],
            task_templates=[
                get_task(TaskType.tabular_regression)(
                    value_column="true_value"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        split_gens = []
        if private_utils.has_private_loc():
            test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
            split_gens += [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
                )
            ]
        else:
            logger.warning(
                "Skipping sumeval test set because "
                f"{private_utils.PRIVATE_LOC} is not set"
            )

        return split_gens

    def _generate_examples(self, filepath):
        """Generate sumeval examples."""

        with open(filepath, encoding="utf-8") as json_file:
            raw_data = json.load(json_file)
            for id_, row in enumerate(raw_data['examples']):
                yield id_, row
