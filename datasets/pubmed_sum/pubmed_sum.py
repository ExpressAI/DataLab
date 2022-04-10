# coding=utf-8
# Copyright 2020 The DataLab authors, and the current dataset script contributor.
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
from datalabs.tasks import Summarization

_CITATION = None
_DESCRIPTION = """
 PubMed dataset for summarization.
 From paper: A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents" by A. Cohan et al.
 See: https://aclanthology.org/N18-2097.pdf 
 See: https://github.com/armancohan/long-summarization
"""

_CITATION = """\
    @inproceedings{cohan-etal-2018-discourse,
    title = "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents",
    author = "Cohan, Arman  and
      Dernoncourt, Franck  and
      Kim, Doo Soon  and
      Bui, Trung  and
      Kim, Seokhwan  and
      Chang, Walter  and
      Goharian, Nazli",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2097",
    doi = "10.18653/v1/N18-2097",
    pages = "615--621",
    abstract = "Neural abstractive summarization models have led to promising results in summarizing relatively short documents. We propose the first model for abstractive summarization of single, longer-form documents (e.g., research papers). Our approach consists of a new hierarchical encoder that models the discourse structure of a document, and an attentive discourse-aware decoder to generate the summary. Empirical results on two large-scale datalab of scientific papers show that our model significantly outperforms state-of-the-art models.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


class PubmedSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for PubmedSum."""

    def __init__(self, **kwargs):
        """BuilderConfig for PubmedSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PubmedSumConfig, self).__init__(**kwargs)


class PubmedSumDataset(datalabs.GeneratorBasedBuilder):
    """PubmedSum Dataset."""

    _TRAIN_FILE = "https://huggingface.co/datalab/ccdv/pubmed-summarization/resolve/main/train.zip"
    _VAL_FILE = (
        "https://huggingface.co/datalab/ccdv/pubmed-summarization/resolve/main/val.zip"
    )
    _TEST_FILE = (
        "https://huggingface.co/datalab/ccdv/pubmed-summarization/resolve/main/test.zip"
    )
    BUILDER_CONFIGS = [
        PubmedSumConfig(
            name="section",
            version=datalabs.Version("1.0.0"),
            description="PubMed dataset for summarization, concat sections",
        ),
        PubmedSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="PubMed dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    # "id": datalab.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/armancohan/long-summarization",
            citation=_CITATION,
            task_templates=[
                Summarization(text_column=_ARTICLE, summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(self._TRAIN_FILE) + "/train.txt"
        val_path = dl_manager.download_and_extract(self._VAL_FILE) + "/val.txt"
        test_path = dl_manager.download_and_extract(self._TEST_FILE) + "/test.txt"

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": val_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate PubmedSum examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                """
                'article_id': str,
                'abstract_text': List[str],
                'article_text': List[str],
                'section_names': List[str],
                'sections': List[List[str]]
                """
                if self.config.name == "document":
                    article = data["article_text"]
                else:
                    article = [
                        item.strip() for sublist in data["sections"] for item in sublist
                    ]
                abstract = data["abstract_text"]
                yield id_, {"text": " ".join(article), "summary": " ".join(abstract)}
