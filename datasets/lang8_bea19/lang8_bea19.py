# coding=utf-8
# Copyright 2022 The TensorFlow datasets Authors and the HuggingFace datasets, DataLab Authors.
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


import datalabs
from datalabs import get_task, TaskType
import os

logger = datalabs.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{mizumoto2012effect,
  title={The effect of learner corpus size in grammatical error correction of ESL writings},
  author={Mizumoto, Tomoya and Hayashibe, Yuta and Komachi, Mamoru and Nagata, Masaaki and Matsumoto, Yuji},
  booktitle={Proceedings of COLING 2012: Posters},
  pages={863--872},
  year={2012}
}
"""

_DESCRIPTION = """\
Lang-8 is an online language learning website which encourages users to correct each other's grammar. The Lang-8 Corpus of Learner English is a somewhat-clean, English subsection of this website"""

_URLS = "https://datalab-hub.s3.amazonaws.com/grammatical_error_correction/lang8.bea19/lang8.bea19.zip"


class Lang8Bea19Config(datalabs.BuilderConfig):
    """BuilderConfig for lang8bea19."""

    def __init__(self, **kwargs):
        """BuilderConfig for lang8bea19.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Lang8Bea19Config, self).__init__(**kwargs)


class Lang8Bea19(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        Lang8Bea19Config(
            name="plain_text",
            version=datalabs.Version("0.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "original": datalabs.Value("string"),
                    "correct":
                        {
                            "m2": datalabs.features.Sequence(datalabs.Value("string")),
                            "sentence": datalabs.Value("string"),
                        }

                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://lang-8.com/",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.grammatical_error_correction_m2)(

                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={
                "filepath": os.path.join(downloaded_files, f"lang8.bea19/lang8.train.auto.bea19.m2")}),

        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            sentences = f.read().strip().split("\n\n")
            skip = {"noop", "UNK", "Um"}
            for sent in sentences:
                sent = sent.split("\n")
                cor_sent = sent[0].split()[1:]  # Ignore "S "
                origin_sent = sent[0].split()[1:]
                m2 = sent[1:]
                edits = sent[1:]
                offset = 0
                for edit in edits:
                    edit = edit.split("|||")
                    if edit[1] in skip: continue  # Ignore certain edits
                    coder = int(edit[-1])
                    if coder != id: continue  # Ignore other coders
                    span = edit[0].split()[1:]  # Ignore "A "
                    start = int(span[0])
                    end = int(span[1])
                    cor = edit[2].split()
                    cor_sent[start + offset:end + offset] = cor
                    offset = offset - (end - start) + len(cor)
                yield key, {
                    "original": " ".join(origin_sent),
                    "correct": {
                        'm2': m2,
                        'sentence': " ".join(cor_sent)
                    }
                }
                key += 1
