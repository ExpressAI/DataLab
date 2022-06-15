"""CiteSum: Citation Text-guided Scientific Extreme Summarization and Low-resource Domain Adaptation"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
TO BE DONE
"""

_DESCRIPTION = """\
We create a large-scale scientific extreme summarization benchmark CiteSum, 
which is automatically derived from citation texts and around
30 times larger than the previous human-annotated dataset SciTLDR.
see: https://arxiv.org/pdf/2205.06207.pdf
"""

_HOMEPAGE = "https://github.com/morningmoni/CiteSum"
_ARTICLE = "text"
_ABSTRACT = "summary"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class CiteSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for CiteSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for CiteSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CiteSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class CiteSumDataset(datalabs.GeneratorBasedBuilder):
    """CiteSum Dataset."""

    _FILE_ID = "1ndHCREXGSPnDUNllladh9qCtayqbXAfJ"

    BUILDER_CONFIGS = [
        CiteSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description=" Scientific extreme summarization dataset.",
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
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "train.json")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": os.path.join(f_path, "val.json")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": os.path.join(f_path, "test.json")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate CiteSum examples."""

        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            line = line.strip()
            data = json.loads(line)
            text = data["src"].strip()
            summary = data["tgt"].strip()
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
