# coding=utf-8
# Copyright 2022 The HuggingFace datasets Authors, DataLab Authors and the current dataset script contributor.
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
"""XSum dataset."""


import json
import os

import datalabs
from datalabs.tasks import Summarization

_CITATION = """
@inproceedings{narayan-etal-2018-dont,
    title = "Don{'}t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    author = "Narayan, Shashi  and
      Cohen, Shay B.  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1206",
    doi = "10.18653/v1/D18-1206",
    pages = "1797--1807",
    abstract = "We introduce {``}extreme summarization{''}, a new single-document summarization task which does not favor extractive strategies and calls for an abstractive modeling approach. The idea is to create a short, one-sentence news summary answering the question {``}What is the article about?{''}. We collect a real-world, large-scale dataset for this task by harvesting online articles from the British Broadcasting Corporation (BBC). We propose a novel abstractive model which is conditioned on the article{'}s topics and based entirely on convolutional neural networks. We demonstrate experimentally that this architecture captures long-range dependencies in a document and recognizes pertinent content, outperforming an oracle extractive system and state-of-the-art abstractive approaches when evaluated automatically and by humans.",
}
"""

_DESCRIPTION = """
Extreme Summarization (XSum) Dataset.

There are three features:
  - document: Input news article.
  - summary: One sentence summary of the article.
  - id: BBC ID of the article.

"""

# From https://github.com/EdinburghNLP/XSum/issues/12
_URL_DATA = "http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz"
_URL_SPLITS = (
    "https://raw.githubusercontent.com/EdinburghNLP/XSum/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
)

_DOCUMENT = "text"
_SUMMARY = "summary"
_ID = "id"

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)


class Xsum(datalabs.GeneratorBasedBuilder):
    """Extreme Summarization (XSum) Dataset."""

    # Version 1.2.0 expands coverage, includes ids, and removes web contents.
    VERSION = datalabs.Version("1.2.0")

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _DOCUMENT: datalabs.Value("string"),
                    _SUMMARY: datalabs.Value("string"),
                    _ID: datalabs.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset",
            citation=_CITATION,
            task_templates=[Summarization(
                text_column=_DOCUMENT,
                summary_column=_SUMMARY),
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        files_to_download = {"data": _URL_DATA, "splits": _URL_SPLITS}
        downloaded_files = dl_manager.download(files_to_download)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "train",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "validation",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "split_path": downloaded_files["splits"],
                    "split_name": "test",
                    "data_dir": "bbc-summary-data",
                    "files": dl_manager.iter_archive(downloaded_files["data"]),
                },
            ),
        ]

    def _generate_examples(self, split_path, split_name, data_dir, files):
        """Yields examples."""

        with open(split_path, "r", encoding="utf-8") as f:
            split_ids = json.load(f)
        split_ids = {k: set(v) for k, v in split_ids.items()}

        for path, f in files:
            if not split_ids[split_name]:
                break
            elif path.startswith(data_dir) and path.endswith(".summary"):
                i = os.path.basename(path).split(".")[0]
                if i in split_ids[split_name]:
                    split_ids[split_name].remove(i)
                    text = "".join(
                        [
                            line.decode("utf-8")
                            for line in f.readlines()
                            if line.decode("utf-8") not in _REMOVE_LINES and line.strip()
                        ]
                    )
                    # Each file follows below format:
                    # [SN]URL[SN]
                    # http://somelink
                    #
                    # [SN]TITLE[SN]
                    # some intro
                    #
                    # [SN]FIRST-SENTENCE[SN]
                    # some intro
                    #
                    # [SN]RESTBODY[SN]
                    # text line.
                    # another text line.
                    # "another text line."

                    # According to the following issue, FIRST-SENTENCE
                    # is the reference summary and TITLE is unused:
                    # https://github.com/EdinburghNLP/XSum/issues/22
                    segs = text.split("[SN]")
                    yield i, {_DOCUMENT: segs[8].strip(), _SUMMARY: segs[6].strip(), _ID: i}
