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



import json
import os
import datalabs
from datalabs.tasks import SequenceLabeling
from datalabs.task_dataset import SequenceLabelingDataset
logger = datalabs.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{emerson-2005-second,
    title = "The Second International {C}hinese Word Segmentation Bakeoff",
    author = "Emerson, Thomas",
    booktitle = "Proceedings of the Fourth {SIGHAN} Workshop on {C}hinese Language Processing",
    year = "2005",
    url = "https://aclanthology.org/I05-3017",
}
"""
_DESCRIPTION = """\
The Simplified Chinese corpora were provided by Beijing University and Microsoft Research Beijing.
"""
_HOMEPAGE = "http://sighan.cs.uchicago.edu/bakeoff2005/"
_LICENSE = "Available for research use"
_URL = "https://datalab-hub.s3.amazonaws.com/msr.zip"







class MSR(datalabs.GeneratorBasedBuilder):


    VERSION = datalabs.Version("1.0.0")



    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "tokens": datalabs.Sequence(datalabs.Value("string")),
                    "tags": datalabs.Sequence(
                        datalabs.features.ClassLabel(
                            names=["B","M","E","S"]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            languages=['zh'],
            version=self.VERSION,
            task_templates=[SequenceLabeling(task="chinese-word-segmentation",
                                             tokens_column="tokens",
                                             tags_column="tags")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/msr/", "train-msr.tsv"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/msr/", "test-msr.tsv"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/msr/", "dev-msr.tsv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    # New sentence
                    if not current_tokens:
                        # Consecutive empty lines will cause empty sentences
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    sentence = (
                        sentence_counter,
                        {
                            "id": str(sentence_counter),
                            "tokens": current_tokens,
                            "tags": current_labels,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                yield sentence_counter, {
                    "id": str(sentence_counter),
                    "tokens": current_tokens,
                    "tags": current_labels,
                }
