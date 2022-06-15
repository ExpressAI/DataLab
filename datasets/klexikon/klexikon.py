"""Klexikon: A German Dataset for Joint Summarization and Simplification"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{Aumiller2022KlexikonAG,
  title={Klexikon: A German Dataset for Joint Summarization and Simplification},
  author={Dennis Aumiller and Michael Gertz},
  journal={ArXiv},
  year={2022},
  volume={abs/2201.07198}
}
"""

_DESCRIPTION = """\
We describe the creation of a new dataset for joint Text Simplification and Summarization 
based on German Wikipedia and the German children’s lexicon "Klexikon", consisting of almost 2,900 documents.
see: https://arxiv.org/pdf/2201.07198.pdf
"""

_HOMEPAGE = "https://github.com/dennlinger/klexikon"
_ARTICLE = "text"
_LICENSE = "MIT License"
_ABSTRACT = "summary"


class KlexikonConfig(datalabs.BuilderConfig):
    """BuilderConfig for Klexikon."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Klexikon.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(KlexikonConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class KlexikonDataset(datalabs.GeneratorBasedBuilder):
    """Klexikon Dataset."""

    BUILDER_CONFIGS = [
        KlexikonConfig(
            name="document",
            version=datalabs.Version("2.0.0"),
            description="A German summarization dataset.",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

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
            license=_LICENSE,
            languages=[
                "de"
            ],  # https://huggingface.co/languages#:~:text=840-,German,-de
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        train_f_path = dl_manager.download(
            "https://huggingface.co/datasets/dennlinger/klexikon/resolve/main/data/train.json"
        )
        valid_f_path = dl_manager.download(
            "https://huggingface.co/datasets/dennlinger/klexikon/resolve/main/data/validation.json"
        )
        test_f_path = dl_manager.download(
            "https://huggingface.co/datasets/dennlinger/klexikon/resolve/main/data/test.json"
        )

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
        """Generate Klexikon examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            data = json.loads(line)
            text = " ".join(data["wiki_sentences"])
            summary = " ".join(data["klexikon_sentences"])
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
