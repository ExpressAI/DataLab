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


import csv

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """\
Text classification aims to assign multiple labels to the sentence. We use the cMedTC dataset, which consists of biomedical texts with multiple labels.
"""

_CITATION = """\
@article{zhang2020conceptualized,
  title={Conceptualized Representation Learning for Chinese Biomedical Text Mining},
  author={Zhang, Ningyu and Jia, Qianghuai and Yin, Kangping and Dong, Liang and Gao, Feng and Hua, Nengwei},
  journal={arXiv preprint arXiv:2008.10813},
  year={2020}
}
"""

_TRAIN_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/cMedTC/train.txt"
)
_VALIDATION_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/cMedTC/dev.txt"
)
_TEST_DOWNLOAD_URL = (
    "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text-classification/cMedTC/test.txt"
)


class cMedTC(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=[
                            "Age",
                            "Disease",
                            "Device",
                            "Pregnancy-related Activity",
                            "Multiple",
                            "Diagnostic",
                            "Risk Assessment",
                            "Organ or Tissue Status",
                            "Allergy Intolerance",
                            "Therapy or Surgery",
                            "Laboratory Examinations",
                            "Addictive Behavior",
                            "Sign",
                            "Pharmaceutical Substance or Drug",
                            "Encounter",
                            "Compliance with Protocol",
                            "Consent",
                            "Enrollment in other studies",
                            "Special Patient Characteristic",
                            "Capacity",
                            "Exercise",
                            "Researcher Decision",
                            "Life Expectancy",
                            "Symptom",
                            "Oral related",
                            "Address",
                            "Smoking Status",
                            "Data Accessible",
                            "Ethnicity",
                            "Literacy",
                            "Non-Neoplasm Disease Stage",
                            "Receptor Status",
                            "Diet",
                            "Education",
                            "Alcohol Consumer",
                            "Healthy",
                            "Neoplasm Status",
                            "Nursing",
                            "Sexual related",
                            "Blood Donation",
                            "Disabilities",
                            "Gender",
                            "Ethical Audit",
                            "Bedtime",
                        ]
                    ),
                }
            ),
            homepage="https://github.com/alibaba-research/ChineseBLUE",
            citation=_CITATION,
            languages=["zh"],
            task_templates=[
                get_task(TaskType.text_classification)(
                    text_column="text", label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:

            for id_, row in enumerate(f):
                label, text = row.strip().split("\t")

                yield id_, {"text": text, "label": label}
