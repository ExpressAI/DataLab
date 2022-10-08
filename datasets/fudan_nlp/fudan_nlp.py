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
import textwrap
import datalabs
from datalabs import get_task, TaskType
from datalabs.utils import private_utils

_DESCRIPTION = """\
There are datasets from Fudan NLP course.
* 2022 Version
"""

_CITATION = """\
"""

_PRIVATE_PREFIX = f"{private_utils.PRIVATE_LOC}/fudan_nlp"

class FudanNlpConfig(datalabs.BuilderConfig):

    def __init__(
        self,
        features,
        data_url,
        data_dir,
        citation,
        url,
        task_templates=None,
        **kwargs,
    ):

        super(FudanNlpConfig, self).__init__(
            version=datalabs.Version("1.0.0", "For Semester 2022"), **kwargs
        )
        self.features = features
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url

        self.task_templates = task_templates

class FudanNlp(datalabs.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        FudanNlpConfig(
            name="movie_review",
            description=textwrap.dedent(
                """\
                Movie-review data for use in sentiment-analysis experiments. Available are collections 
                 of movie-review documents labeled with respect to their overall sentiment polarity (positive or negative)
                  or subjective rating (e.g., "two and a half stars")
                """
            ),
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["positive", "negative"]
                    ),
                }
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/fudan_nlp/mr.zip",
            data_dir="mr",
            citation=textwrap.dedent(
                """\
                    @inproceedings{pang-lee-2005-seeing,
                        title = "Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales",
                        author = "Pang, Bo  and
                          Lee, Lillian",
                        booktitle = "Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics ({ACL}{'}05)",
                        month = jun,
                        year = "2005",
                        address = "Ann Arbor, Michigan",
                        publisher = "Association for Computational Linguistics",
                        url = "https://aclanthology.org/P05-1015",
                        doi = "10.3115/1219840.1219855",
                        pages = "115--124",
                    }
                            """
            ),
            url="http://www.cs.cornell.edu/people/pabo/movie-review-data/",
            task_templates=[
                get_task(TaskType.sentiment_classification)(
                    text_column="text", label_column="label"
                )
            ],
        ),
        FudanNlpConfig(
            name="conll2003",
            description=textwrap.dedent(
                """\
                Large Movie Review Dataset.
                This is a dataset for binary sentiment classification containing substantially \
                more data than previous benchmark datasets.\
                """
            ),
            features=datalabs.Features(
                {
                    "tokens": datalabs.Sequence(datalabs.Value("string")),
                    "tags": datalabs.Sequence(
                        datalabs.features.ClassLabel(names=[
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                            ])
                    ),
                }
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/fudan_nlp/conll2003_v2.zip",
            data_dir="conll2003",
            citation=textwrap.dedent(
                """\
                    @inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
                        title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
                        author = "Tjong Kim Sang, Erik F.  and
                          De Meulder, Fien",
                        booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
                        year = "2003",
                        url = "https://www.aclweb.org/anthology/W03-0419",
                        pages = "142--147",
                    }
                            """
            ),
            url="https://www.clips.uantwerpen.be/conll2003/ner/",
            task_templates=[
                get_task(TaskType.named_entity_recognition)(
                    tokens_column="tokens", tags_column="tags"
                )
            ],
        ),
    ]


    def _info(self):

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _DESCRIPTION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        dataset_path = dl_manager.download_and_extract(self.config.data_url)
        config_name = self.config.name if \
            self.config.name != "movie_review" else "mr"
        dataset_path_test =  dl_manager.download_and_extract(
            f'{_PRIVATE_PREFIX}/{config_name}/test.data')


        split_gens = [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{dataset_path}/{config_name}/train.data"
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": f"{dataset_path}/{config_name}/valid.data"
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": dataset_path_test
                },
            ),
        ]

        return split_gens


    def _generate_examples(self, filepath):

        if self.config.name == "movie_review":
            textualize_label = {"0": "negative", "1": "positive"}
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter="\t")
                for id_, row in enumerate(csv_reader):
                    text, label = row[0], row[1]

                    label = textualize_label[label] if label in textualize_label.keys()\
                        else label
                    text = text

                    raw_feature_info = {"text": text, "label": label}

                    yield id_, raw_feature_info

        # for the dataset: conll2003
        elif self.config.name == "conll2003":
            with open(filepath, encoding="utf-8") as f:
                guid = 0
                tokens = []
                tags = []
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if tokens:
                            yield guid, {
                                "tokens": tokens,
                                "tags": tags,
                            }
                            guid += 1
                            tokens = []
                            tags = []
                    else:
                        # conll2003 tokens are space separated
                        splits = line.split(" ")
                        tokens.append(splits[0])
                        tags.append(splits[3].rstrip())

                # last example
                if len(tokens) != 0:
                    yield guid, {
                        "tokens": tokens,
                        "tags": tags,
                    }

