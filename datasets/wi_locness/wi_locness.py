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
@inproceedings{bryant-etal-2019-bea,
    title = "The {BEA}-2019 Shared Task on Grammatical Error Correction",
    author = "Bryant, Christopher  and
      Felice, Mariano  and
      Andersen, {\O}istein E.  and
      Briscoe, Ted",
    booktitle = "Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-4406",
    doi = "10.18653/v1/W19-4406",
    pages = "52--75",
    abstract = "This paper reports on the BEA-2019 Shared Task on Grammatical Error Correction (GEC). As with the CoNLL-2014 shared task, participants are required to correct all types of errors in test data. One of the main contributions of the BEA-2019 shared task is the introduction of a new dataset, the Write{\&}Improve+LOCNESS corpus, which represents a wider range of native and learner English levels and abilities. Another contribution is the introduction of tracks, which control the amount of annotated data available to participants. Systems are evaluated in terms of ERRANT F{\_}0.5, which allows us to report a much wider range of performance statistics. The competition was hosted on Codalab and remains open for further submissions on the blind test set.",
}
"""

_DESCRIPTION = """\
WI-LOCNESS is part of the Building Educational Applications 2019 Shared Task for Grammatical Error Correction. It consists of two datasets:

LOCNESS: is a corpus consisting of essays written by native English students.
Cambridge English Write & Improve (W&I): Write & Improve (Yannakoudakis et al., 2018) is an online web platform that assists non-native English students with their writing. Specifically, students from around the world submit letters, stories, articles and essays in response to various prompts, and the W&I system provides instant feedback. Since W&I went live in 2014, W&I annotators have manually annotated some of these submissions and assigned them a CEFR level."""

_URLS = "https://datalab-hub.s3.amazonaws.com/grammatical_error_correction/wi%2Blocness/wi%2Blocness.zip"


class WiLocnessConfig(datalabs.BuilderConfig):
    """BuilderConfig for wilocness"""

    def __init__(self, **kwargs):
        """BuilderConfig for wilocness.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WiLocnessConfig, self).__init__(**kwargs)


class Wi_Locness(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        WiLocnessConfig(
            name="plain_text",
            version=datalabs.Version("2.1.0", ""),
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
            homepage="https://www.cl.cam.ac.uk/research/nl/bea2019st/",
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
                "filepath": os.path.join(downloaded_files, f"wi+locness/ABC.train.gold.bea19.m2")}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={
                "filepath": os.path.join(downloaded_files, f"wi+locness/ABCN.dev.gold.bea19.m2")}),

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
