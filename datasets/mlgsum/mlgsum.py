"""MLGSum: multiligual summarization dataset"""
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{wang-etal-2021-contrastive,
    title = "Contrastive Aligned Joint Learning for Multilingual Summarization",
    author = "Wang, Danqing  and
      Chen, Jiaze  and
      Zhou, Hao  and
      Qiu, Xipeng  and
      Li, Lei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.242",
    doi = "10.18653/v1/2021.findings-acl.242",
    pages = "2739--2750",
}
"""

_DESCRIPTION = """\
MLGSum contins 12 languages for the multilingual summarization task.
Articles are collectde from news websites with multiple languages, such as BBC and france24, and faz.
The brief introduction written by editors are used as summaries.
see: https://aclanthology.org/2021.findings-acl.242.pdf
"""

_HOMEPAGE = "https://github.com/dqwang122/CALMS"
_ABSTRACT = "summary"
_ARTICLE = "text"
_CLEAN_TAR_GZ_URL = "https://storage.googleapis.com/inspired-public-data/datasets/mlgsum/raw/clean.tar.gz"

class MLGSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for MLGSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MLGSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MLGSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MLGSumDataset(datalabs.GeneratorBasedBuilder):
    """MLGSum Dataset."""

    _FILE_ID = "1ZoOdEIDBGuG7ucdkjnGr5UVQRUkrE1pm"
    _LANG = ["de", "en", "es", "fr", "hi", "id", "pt", "ru", "tr", "uk", "vi", "zh"]
    BUILDER_CONFIGS = list(
        [
            MLGSumConfig(
                name=l,
                version=datalabs.Version("1.0.0"),
                description=f"MLGSum Dataset for multiligual summarization, {l} split",
                task_templates=[
                    get_task(TaskType.summarization)(
                        source_column=_ARTICLE, reference_column=_ABSTRACT
                    )
                ],
            )
            for l in [
                "de",
                "en",
                "es",
                "fr",
                "hi",
                "id",
                "pt",
                "ru",
                "tr",
                "uk",
                "vi",
                "zh",
            ]
        ]
    )
    DEFAULT_CONFIG_NAME = "en"

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
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_CLEAN_TAR_GZ_URL)
        lang_id = self.config.name
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "text_path": os.path.join(
                        f_path, f"./clean/{lang_id}/train.{lang_id}.doc"
                    ),
                    "summary_path": os.path.join(
                        f_path, f"./clean/{lang_id}/train.{lang_id}.sum"
                    ),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "text_path": os.path.join(
                        f_path, f"./clean/{lang_id}/dev.{lang_id}.doc"
                    ),
                    "summary_path": os.path.join(
                        f_path, f"./clean/{lang_id}/dev.{lang_id}.sum"
                    ),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "text_path": os.path.join(
                        f_path, f"./clean/{lang_id}/test.{lang_id}.doc"
                    ),
                    "summary_path": os.path.join(
                        f_path, f"./clean/{lang_id}/test.{lang_id}.sum"
                    ),
                },
            ),
        ]

    def _generate_examples(self, text_path, summary_path):
        """Generate MLGSum examples."""
        with open(text_path, encoding="utf-8") as f_src, open(
            summary_path, encoding="utf-8"
        ) as f_tgt:
            for (id_, (x, y)) in enumerate(zip(f_src, f_tgt)):
                x = x.strip()
                y = y.strip()
                x = x.replace("<q>", " ")
                y = y.replace("<q>", " ")
                yield id_, {_ARTICLE: x, _ABSTRACT: y}
