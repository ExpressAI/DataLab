# coding=utf-8
# Copyright 2021 The HuggingFace datalab Authors and the current dataset script contributor.
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
import os

import datalabs
from datalabs.tasks import TextMatching

_CITATION = """\
@inproceedings{marelli-etal-2014-sick,
    title = "A {SICK} cure for the evaluation of compositional distributional semantic models",
    author = "Marelli, Marco  and
      Menini, Stefano  and
      Baroni, Marco  and
      Bentivogli, Luisa  and
      Bernardi, Raffaella  and
      Zamparelli, Roberto",
    booktitle = "Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)",
    month = may,
    year = "2014",
    address = "Reykjavik, Iceland",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf",
    pages = "216--223",
}
"""

_DESCRIPTION = """\
Shared and internationally recognized benchmarks are fundamental for the development of any computational system.
We aim to help the research community working on compositional distributional semantic models (CDSMs) by providing SICK (Sentences Involving Compositional Knowldedge), a large size English benchmark tailored for them.
SICK consists of about 10,000 English sentence pairs that include many examples of the lexical, syntactic and semantic phenomena that CDSMs are expected to account for, but do not require dealing with other aspects of existing sentential data sets (idiomatic multiword expressions, named entities, telegraphic language) that are not within the scope of CDSMs.
By means of crowdsourcing techniques, each pair was annotated for two crucial semantic tasks: relatedness in meaning (with a 5-point rating scale as gold score) and entailment relation between the two elements (with three possible gold labels: entailment, contradiction, and neutral).
The SICK data set was used in SemEval-2014 Task 1, and it freely available for research purposes.
"""

_DOWNLOAD_URL = "https://zenodo.org/record/2787612/files/SICK.zip?download=1"


class SICK(datalabs.GeneratorBasedBuilder):
    """The SICK (Sentences Involving Compositional Knowldedge) dataset."""

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "text1": datalabs.Value("string"),
                    "text2": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "relatedness_score": datalabs.Value("float"),
                    "entailment_AB": datalabs.Value("string"),
                    "entailment_BA": datalabs.Value("string"),
                    "sentence_A_original": datalabs.Value("string"),
                    "sentence_B_original": datalabs.Value("string"),
                    "sentence_A_dataset": datalabs.Value("string"),
                    "sentence_B_dataset": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="http://marcobaroni.org/composes/sick.html",
            citation=_CITATION,
            task_templates=[TextMatching(
                text1_column="text1",
                text2_column="text2",
                task="natural-language-inference",
                label_column="label"),
            ],
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(dl_dir, "SICK.txt"), "key": "TRAIN"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(dl_dir, "SICK.txt"), "key": "TRIAL"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"filepath": os.path.join(dl_dir, "SICK.txt"), "key": "TEST"},
            ),
        ]

    def _generate_examples(self, filepath, key):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = [s.strip() for s in line.split("\t")]
                if data[-1] == key:
                    yield data[0], {
                        "id": data[0],
                        "text1": data[1],
                        "text2": data[2],
                        "label": data[3].lower(),
                        "relatedness_score": data[4],
                        "entailment_AB": data[5],
                        "entailment_BA": data[6],
                        "sentence_A_original": data[7],
                        "sentence_B_original": data[8],
                        "sentence_A_dataset": data[9],
                        "sentence_B_dataset": data[10],
                    }
