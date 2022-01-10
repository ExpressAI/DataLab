# coding=utf-8
# Copyright 2020 The TensorFlow datalab Authors and the HuggingFace datalab Authors.
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

# Lint as: python3
"""AG News topic classification dataset."""


import csv

import datalab
from datalab.tasks import TextClassification


_DESCRIPTION = """\
AG is a collection of more than 1 million news articles. News articles have been
gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
activity. ComeToMyHead is an academic news search engine which has been running
since July, 2004. The dataset is provided by the academic comunity for research
purposes in data mining (clustering, classification, etc), information retrieval
(ranking, search, etc), xml, data compression, data streaming, and any other
non-commercial activity. For more information, please refer to the link
http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang
(xiang.zhang@nyu.edu) from the dataset above. It is used as a text
classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
LeCun. Character-level Convolutional Networks for Text Classification. Advances
in Neural Information Processing Systems 28 (NIPS 2015).
"""

_CITATION = """\
@inproceedings{Zhang2015CharacterlevelCN,
  title={Character-level Convolutional Networks for Text Classification},
  author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  booktitle={NIPS},
  year={2015}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


class AGNews(datalab.GeneratorBasedBuilder):
    """AG News topic classification dataset."""

    def _info(self):
        return datalab.DatasetInfo(
            description=_DESCRIPTION,
            features=datalab.Features(
                {
                    "text": datalab.Value("string"),
                    "label": datalab.features.ClassLabel(names=["World", "Sports", "Business", "Science and Technology"]),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalab.SplitGenerator(name=datalab.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalab.SplitGenerator(name=datalab.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate AG News examples."""

        textualize_label = {"1":"World",
                                 "2":"Sports",
                                 "3":"Business",
                                 "4":"Science and Technology"}


        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                label, title, description = row
                # Original labels are [1, 2, 3, 4] ->
                #                   ['World', 'Sports', 'Business', 'Sci/Tech']
                # Re-map to [0, 1, 2, 3].
                label = textualize_label[label]
                text = " ".join((title, description))
                yield id_, {"text": text, "label": label}
