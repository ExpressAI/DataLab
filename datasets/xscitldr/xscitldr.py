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


"""X-SCITLDR abstractive summarization dataset."""
import json
import os
import datalabs
from datalabs import get_task, TaskType


_CITATION = """\
@article{Takeshita2022XSCITLDRCE,
  title={X-SCITLDR: Cross-Lingual Extreme Summarization of Scholarly Documents},
  author={Sotaro Takeshita and Tommaso Green and Niklas Friedrich and K. Eckert and Simone Paolo Ponzetto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.15051}
}
"""
_DESCRIPTION = """\
we present an abstractive cross-lingual summarization dataset for four different languages in the scholarly domain, 
which enables us to train and evaluate models that process English papers and generate summaries in German, Italian, Chinese and Japanese.
"""
_HOMEPAGE = "https://github.com/sobamchan/xscitldr"
_ABSTRACT = "summary"
_ARTICLE = "text"


class XSCITLDRConfig(datalabs.BuilderConfig):
    """BuilderConfig for XSCITLDR."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for XSCITLDR.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(XSCITLDRConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class XSCITLDRDataset(datalabs.GeneratorBasedBuilder):
    """XSCITLDR Dataset."""
    _LANG = ["ja", "de", "zh", "it"]
    _URLs = {
        "en-ja_train": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/human/ja/train.jsonl",
        "en-ja_test": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/human/ja/test.jsonl",
        "en-ja_val": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/human/ja/val.jsonl",
        "en-de_train": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/de/train.jsonl",
        "en-de_test": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/de/test.jsonl",
        "en-de_val": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/de/val.jsonl",
        "en-zh_train": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/zh/train.jsonl",
        "en-zh_test": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/zh/test.jsonl",
        "en-zh_val": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/zh/val.jsonl",
        "en-it_train": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/it/train.jsonl",
        "en-it_test": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/it/test.jsonl",
        "en-it_val": "https://raw.githubusercontent.com/sobamchan/xscitldr/main/data/postedit/it/val.jsonl",
    }
    BUILDER_CONFIGS = list([
        XSCITLDRConfig(
            name=f"en-{l}",
            version=datalabs.Version("1.0.0"),
            description=f"XSCITLDR Dataset for crosslingual summarization, en-{l} split",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ) for l in ["ja", "de", "zh", "it"]

    ])
    DEFAULT_CONFIG_NAME = "en-de"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        lang_id = self.config.name
        train_path = dl_manager.download(self._URLs[f"{lang_id}_train"])
        test_path = dl_manager.download(self._URLs[f"{lang_id}_test"])
        val_path = dl_manager.download(self._URLs[f"{lang_id}_val"])
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": train_path,
                    }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": val_path,
                    }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": test_path,
                    }
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate XSCITLDR examples."""
        with open(f_path, encoding="utf-8") as f: 
            for (id_, x) in enumerate(f):
                x = json.loads(x)
                yield id_, {_ARTICLE: x["source"].strip(), _ABSTRACT: " ".join(x["target"]).strip()}
                