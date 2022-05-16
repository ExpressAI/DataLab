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

import csv
import datalabs
from datalabs import get_task, TaskType


_DESCRIPTION = """\
The evaluation object extraction task aims to automatically extract the evaluation objects contained in a given review text. 
This task is one of the basic tasks in sentiment analysis, and the dataset covers data scraped on Baidu, Dianping, and Mafengwo.
For more information, please refer to https://www.luge.ai/#/luge/dataDetail?id=19. 
"""

_CITATION = """\
@InProceedings{pmlr-v95-li18d,
  title = 	 {Character-based BiLSTM-CRF Incorporating POS and Dictionaries for Chinese Opinion Target Extraction},
  author =       {Li, Yanzeng and Liu, Tingwen and Li, Diying and Li, Quangang and Shi, Jinqiao and Wang, Yanqiu},
  booktitle = 	 {Proceedings of The 10th Asian Conference on Machine Learning},
  pages = 	 {518--533},
  year = 	 {2018},
  editor = 	 {Zhu, Jun and Takeuchi, Ichiro},
  volume = 	 {95},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {14--16 Nov},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v95/li18d/li18d.pdf},
  url = 	 {https://proceedings.mlr.press/v95/li18d.html},
  abstract = 	 {Opinion target extraction (OTE) is a fundamental step for sentiment analysis and opinion summarization. We analyze the difference between Chinese and the Indo-European languages family, and reduce Chinese OTE to a character-based sequence tagging task. Then we introduce two novel features for each character by distributing POS differentially and using predefined templates over contexts and dictionaries. We further propose a character-based BiLSTM-CRF model incorporating the two feature sequences aligned with the character sequence. Experimental results on real-world consumer review datasets show that our work significantly outperforms the baseline methods for Chinese OTE.}
}
"""

_LICENSE = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/cote/COTE-BD/License.pdf"

_TRAIN_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/cote/COTE-BD/train.tsv"
# _TEST_DOWNLOAD_URL = "http://cdatalab1.oss-cn-beijing.aliyuncs.com/conditional_generation/cote/COTE-BD/test.tsv"

_OPINION = "opinion"
_TARGET = "target"


class COTE_BD(datalabs.GeneratorBasedBuilder):

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _OPINION: datalabs.Value("string"),
                    _TARGET: datalabs.Value("string"),
                }
            ),
            supervised_keys=(_OPINION, _TARGET),
            homepage="https://proceedings.mlr.press/v95/li18d.html",
            citation=_CITATION,
            task_templates=[get_task(TaskType.opinion_target_extraction)(
                source_column=_OPINION,
                reference_column=_TARGET),
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath=None):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter = '\t')
            header = 0
            for id_, line in enumerate(csv_reader):
                if header > 0:
                    _TARGET, _OPINION = line
                    yield id_, {'opinion': _OPINION, 'target': _TARGET}
                header = header + 1

