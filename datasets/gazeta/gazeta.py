"""Gazeta: Dataset for Automatic Summarization of Russian News."""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@InProceedings{Gusev2020gazeta,
    author="Gusev, Ilya",
    title="Dataset for Automatic Summarization of Russian News",
    booktitle="Artificial Intelligence and Natural Language",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="{122--134}",
    isbn="978-3-030-59082-6",
    doi={10.1007/978-3-030-59082-6_9}
}
"""

_DESCRIPTION = """\
We introduce the first Russian summarization dataset in the news domain
see: https://arxiv.org/pdf/2006.11063.pdf
"""

_HOMEPAGE = "https://github.com/IlyaGusev/gazeta"
_ABSTRACT = "summary"
_ARTICLE = "text"


# def _gdrive_url(id):
#     return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class GazetaConfig(datalabs.BuilderConfig):
    """BuilderConfig for Gazeta."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Gazeta.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GazetaConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class GazetaDataset(datalabs.GeneratorBasedBuilder):
    """Gazeta Dataset."""

    _FILE_ID = "https://www.dropbox.com/s/lb50mk5jujjjqbi/gazeta_jsonl_v2.tar.gz?dl=1"

    BUILDER_CONFIGS = [
        GazetaConfig(
            name="document",
            version=datalabs.Version("2.0.0"),
            description="Dataset for Automatic Summarization of Russian News.",
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
            languages=[
                "ru"
            ],  # https://huggingface.co/languages#:~:text=319-,Russian,-ru
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._FILE_ID)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "gazeta_train.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": os.path.join(f_path, "gazeta_val.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": os.path.join(f_path, "gazeta_test.jsonl")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate Gazeta examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            data = json.loads(line)
            article = data["text"]
            summary = data["summary"]
            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: article, _ABSTRACT: summary}
            yield id_, raw_feature_info
