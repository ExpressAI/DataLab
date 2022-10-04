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

_DESCRIPTION = """\
There are datasets from Fudan NLP course.
* 2022 Version
"""

_CITATION = """\
"""

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
            name="imdb",
            description=textwrap.dedent(
                """\
                Large Movie Review Dataset.
                This is a dataset for binary sentiment classification containing substantially \
                more data than previous benchmark datasets.\
                """
            ),
            features = datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(
                        names=["positive", "negative"]
                    ),
                }
            ),
            data_url="https://datalab-hub.s3.amazonaws.com/fudan_nlp/imdb.zip",
            data_dir="imdb",
            citation=textwrap.dedent(
                """\
                @InProceedings{maas-EtAl:2011:ACL-HLT2011,
                  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
                  title     = {Learning Word Vectors for Sentiment Analysis},
                  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
                  month     = {June},
                  year      = {2011},
                  address   = {Portland, Oregon, USA},
                  publisher = {Association for Computational Linguistics},
                  pages     = {142--150},
                  url       = {http://www.aclweb.org/anthology/P11-1015}
                            }"""
            ),
            url="https://datalab.stanford.edu/sentiment/index.html",
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
                                                    "B-ADJP",
                                                    "I-ADJP",
                                                    "B-ADVP",
                                                    "I-ADVP",
                                                    "B-CONJP",
                                                    "I-CONJP",
                                                    "B-INTJ",
                                                    "I-INTJ",
                                                    "B-LST",
                                                    "I-LST",
                                                    "B-NP",
                                                    "I-NP",
                                                    "B-PP",
                                                    "I-PP",
                                                    "B-PRT",
                                                    "I-PRT",
                                                    "B-SBAR",
                                                    "I-SBAR",
                                                    "B-UCP",
                                                    "I-UCP",
                                                    "B-VP",
                                                    "I-VP",
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



        split_gens = [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{dataset_path}/{self.config.name}/train.data"
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": f"{dataset_path}/{self.config.name}/test.data"
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": f"{dataset_path}/{self.config.name}/valid.data"
                },
            ),
        ]

        return split_gens


    def _generate_examples(self, filepath):
        # for the dataset: imdb
        if self.config.name == "imdb":
            textualize_label = {"pos": "positive", "neg": "negative"}
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter="\t")
                for id_, row in enumerate(csv_reader):
                    text, label = row
                    label = textualize_label[label]
                    yield id_, {"text": text, "label": label}

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
                        tags.append(splits[2].rstrip())

                # last example
                if len(tokens) != 0:
                    yield guid, {
                        "tokens": tokens,
                        "tags": tags,
                    }

