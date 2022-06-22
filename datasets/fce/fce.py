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
@inproceedings{yannakoudakis2011new,
  title={A new dataset and method for automatically grading ESOL texts},
  author={Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben},
  booktitle={Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies},
  pages={180--189},
  year={2011}
}
"""

_DESCRIPTION = """\
The Cambridge Learner Corpus First Certificate in English (CLC FCE) dataset consists of short texts, written by learners of English as an additional language in response to exam prompts eliciting free-text answers and assessing mastery of the upper-intermediate proficiency level. The texts have been manually error-annotated using a taxonomy of 77 error types. The full dataset consists of 323,192 sentences. The publicly released subset of the dataset, named FCE-public, consists of 33,673 sentences split into test and training sets of 2,720 and 30,953 sentences, respectively.
"""

_URLS = "https://datalab-hub.s3.amazonaws.com/grammatical_error_correction/fce/fce.zip"


class FceConfig(datalabs.BuilderConfig):
    """BuilderConfig for fce."""

    def __init__(self, **kwargs):
        """BuilderConfig for fce.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FceConfig, self).__init__(**kwargs)


class Fce(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        FceConfig(
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
            supervised_keys=None,
            homepage="https://ilexir.co.uk/datasets/index.html",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.grammatical_error_correction_m2)(

                )
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": os.path.join(downloaded_files,f"fce/fce.train.gold.bea19.m2")}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(downloaded_files,f"fce/fce.dev.gold.bea19.m2")}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST,
                                    gen_kwargs={"filepath": os.path.join(downloaded_files, f"fce/fce.test.gold.bea19.m2")}),
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
                cor = ''
                for edit in edits:
                    edit = edit.split("|||")
                    if edit[1] in skip: continue  # Ignore certain edits
                    coder = int(edit[-1])
                    # if coder != id: continue  # Ignore other coders
                    if coder != 0: continue  # Ignore other coders
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
                        # 'sentence': "edits: "+"@".join(edits)+" cor: " + cor + "offset: "+str(offset)
                        'sentence': " ".join(cor_sent)
                    }
                }
                key += 1
