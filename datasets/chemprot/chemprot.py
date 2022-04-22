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

import json
import datalabs
from datalabs.tasks import TextClassification

_DESCRIPTION = """\
ChemProt is a publicly available compilation of chemical-protein-disease annotation
resources that enables the study of systems pharmacology for a small molecule across
multiple layers of complexity from molecular to clinical levels.
https://pubmed.ncbi.nlm.nih.gov/26876982/

It was curated into a format for text classification by https://arxiv.org/abs/2004.10964
"""

_CITATION = """\
@article{kringelum2016chemprot,
  title={ChemProt-3.0: a global chemical biology diseases mapping},
  author={Kringelum, Jens and Kjaerulff, Sonny Kim and Brunak, S{\o}ren and Lund, Ole and Oprea, Tudor I and Taboureau, Olivier},
  journal={Database},
  volume={2016},
  year={2016},
  publisher={Oxford Academic}
}
"""

_TRAIN_DOWNLOAD_URL = "https://huggingface.co/datasets/zapsdcn/chemprot/raw/main/chemprot_train.jsonl"
_DEV_DOWNLOAD_URL = "https://huggingface.co/datasets/zapsdcn/chemprot/raw/main/chemprot_dev.jsonl"
_TEST_DOWNLOAD_URL = "https://huggingface.co/datasets/zapsdcn/chemprot/raw/main/chemprot_test.jsonl"
_CLASS_LABELS = [
    "ACTIVATOR",
    "AGONIST",
    "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
    "ANTAGONIST",
    "DOWNREGULATOR",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "INHIBITOR",
    "PRODUCT-OF",
    "SUBSTRATE",
    "SUBSTRATE_PRODUCT-OF",
    "UPREGULATOR"
]

class CR(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "text": datalabs.Value("string"),
                    "label": datalabs.features.ClassLabel(names=_CLASS_LABELS),
                }
            ),
            homepage="https://pubmed.ncbi.nlm.nih.gov/26876982/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        print(f"dev_path: \t{dev_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate CR examples."""

        with open(filepath, encoding="utf-8") as jsonl_file:
            for id_, line in enumerate(jsonl_file):
                data = json.loads(line)
                yield id_, {"text": data["text"], "label": data["label"]}
