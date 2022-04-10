# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors and DataLab Authors.
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


"""Dataset config script for ag_news （this code is originally from huggingface, them modified by datalab）"""

import csv
import os
from pathlib import Path

from featurize.general import get_features_sample_level

import datalabs
from datalabs import Dataset, Prompts
from datalabs.tasks import TextClassification, TopicClassification
from datalabs.utils.more_features import get_feature_arguments, prefix_dict_key

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
_PROMPT_URL = "https://raw.githubusercontent.com/ExpressAI/DataLab/main/datasets/ag_news/prompts.json"


# class AGNewsDataset(Dataset):
#     def apply(self, func):
#         if func._type == 'Aggregating':


def instantiate_task_prompt(category_names):
    # instantiate task prompts into dataset prompts
    textual_choices_with_or = (
        ", ".join(category_names[:-1]) + " or " + category_names[-1]
    )
    textual_choices_without_or = ", ".join(category_names)
    category_to_answers = dict(
        zip(category_names, [[category] for category in category_names])
    )
    task_prompts = TopicClassification.get_prompts()

    for prompt_id in task_prompts:
        task_prompts[prompt_id].answers = category_to_answers
        task_prompts[prompt_id].template = (
            task_prompts[prompt_id]
            .template.replace("{{textual_choices_with_or}}", textual_choices_with_or)
            .replace("{{textual_choices_without_or}}", textual_choices_without_or)
        )
    return task_prompts


def infer_schema_dataset_level(sample_level_schema: dict):
    dataset_level_schema = {}
    for feature_name, value in sample_level_schema.items():
        if isinstance(value, int) or isinstance(value, float):
            dataset_level_schema[feature_name] = value
    return dataset_level_schema


EXPAND = True
FIELD = "text"


class AGNews(datalabs.GeneratorBasedBuilder):
    def _info(self):

        category_names = ["World", "Sports", "Business", "Science and Technology"]
        # Task prompts
        prompts = instantiate_task_prompt(
            category_names
        )  # instantiate task prompt based on the current dataset
        features_dataset = {}
        features_sample = datalabs.Features(
            {
                FIELD: datalabs.Value("string"),
                "label": datalabs.features.ClassLabel(names=category_names),
            }
        )

        if self.feature_expanding:
            sample_level_schema = get_features_sample_level("This is a test sample")
            dict_feature_argument = get_feature_arguments(
                sample_level_schema, field=FIELD, feature_level="sample_level"
            )
            additional_features = datalabs.Features(dict_feature_argument)
            features_sample.update(additional_features)

            dataset_level_schema = infer_schema_dataset_level(sample_level_schema)
            dict_feature_argument = get_feature_arguments(
                dataset_level_schema,
                field="avg" + "_" + FIELD,
                feature_level="dataset_level",
            )
            features_dataset = datalabs.Features(dict_feature_argument)

        # Add PromptSource prompts
        prompts.update(Prompts.from_url(_PROMPT_URL))

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                TopicClassification(text_column="text", label_column="label")
            ],
            prompts=prompts,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate AG News examples."""

        # map the label into textual string
        textualize_label = {
            "1": "World",
            "2": "Sports",
            "3": "Business",
            "4": "Science and Technology",
        }

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            # using this for tsv: csv_reader = csv.reader(csv_file, delimiter='\t')
            for id_, row in enumerate(csv_reader):
                label, title, description = row
                label = textualize_label[label]
                text = " ".join((title, description))
                # yield id_, {"text": text, "label": label}

                raw_feature_info = {FIELD: text, "label": label}

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = prefix_dict_key(
                        get_features_sample_level(text), FIELD
                    )
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
