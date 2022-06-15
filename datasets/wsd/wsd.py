"""WSD: A Novel Wikipedia based Dataset for Monolingual and Cross-Lingual Summarization"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{fatima2021,
    title={A Novel Wikipedia based Dataset for Monolingual and Cross-Lingual Summarization},
    author={Mehwish Fatima, Michael Strube},
    booktitle={Proceedings of the 3rd Workshop on New Frontiers in Summarization},
    year={2021}
}
"""

_DESCRIPTION = """\
We present a new dataset for monolingual and cross-lingual summarization considering the English-German pair.
see: https://aclanthology.org/2021.newsum-1.5.pdf
"""

_HOMEPAGE = "https://github.com/MehwishFatimah/wsd"
_ARTICLE = "text"
_ABSTRACT = "summary"


class WSDConfig(datalabs.BuilderConfig):
    """BuilderConfig for WSD."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WSD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WSDConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WSDDataset(datalabs.GeneratorBasedBuilder):
    """WSD Dataset."""

    URL = {
        "monolingual": "https://wsd.h-its.org/wms.zip",
        "crosslingual": "https://wsd.h-its.org/wcls.zip",
    }

    BUILDER_CONFIGS = [
        WSDConfig(
            name="monolingual",
            version=datalabs.Version("1.0.0"),
            description="English-to-English summarization dataset.",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        WSDConfig(
            name="crosslingual",
            version=datalabs.Version("1.0.0"),
            description="English-to-German summarization dataset.",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "crosslingual"

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
            languages=[
                "de",
                "en",
            ],  # https://huggingface.co/languages#:~:text=840-,German,-de
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(self.URL[self.config.name])

        if self.config.name == "monolingual":
            train_f_path = os.path.join(f_path, "en_train.json")
            valid_f_path = os.path.join(f_path, "en_val.json")
            test_f_path = os.path.join(f_path, "en_test.json")
        elif self.config.name == "crosslingual":
            train_f_path = os.path.join(f_path, "en_de_train.json")
            valid_f_path = os.path.join(f_path, "en_de_val.json")
            test_f_path = os.path.join(f_path, "en_de_test.json")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate WSD examples."""

        f = open(f_path, encoding="utf-8")
        dataset = json.load(f)
        texts = dataset["text"]
        if self.config.name == "monolingual":
            summaries = dataset["summary"]
        elif self.config.name == "crosslingual":
            summaries = dataset["dsummary"]

        datas = []
        for text, summary in zip(texts.values(), summaries.values()):
            datas.append((text.strip(), summary.strip()))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
