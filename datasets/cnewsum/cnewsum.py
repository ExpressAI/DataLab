"""CNewSum Chinese news summarization dataset."""
import json
import os

import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@inproceedings{Wang2021CNewSumAL,
  title={CNewSum: A Large-Scale Summarization Dataset with Human-Annotated Adequacy and Deducibility Level},
  author={Danqing Wang and Jiaze Chen and Xianze Wu and Hao Zhou and Lei Li},
  booktitle={NLPCC},
  year={2021}
}
"""

_DESCRIPTION = """\
In this paper, we present a large-scale Chinese news summarization dataset CNewSum, 
which consists of 304,307 documents and human-written summaries for the news feed. 
It has long documents with high abstractive summaries, which can encourage document-level 
understanding and generation for current summarization models.
see: https://arxiv.org/pdf/2110.10874.pdf
"""

_HOMEPAGE = "https://dqwang122.github.io/projects/CNewSum/"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class CNewSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for CNewSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for CNewSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CNewSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class CNewSumDataset(datalabs.GeneratorBasedBuilder):
    """CNewSum Dataset."""

    _FILE_ID = "1A_YcQ3cBAI7u9iVIoCeVLLgwU7UUzHHv"
    BUILDER_CONFIGS = [
        CNewSumConfig(
            name="document",
            version=datalabs.Version("2.0.0"),
            description="CNewSum, A Large-scale Chinese News Summarization Dataset",
            task_templates=[
                Summarization(text_column=_ARTICLE, summary_column=_ABSTRACT)
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
            languages=["zh"],
            task_templates=[
                Summarization(text_column=_ARTICLE, summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "final/train.simple.label.jsonl")
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "final/dev.simple.label.jsonl")
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "final/test.simple.label.jsonl")
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate CNewSum examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()

        datas = []
        for line in lines:
            data = json.loads(line)
            article = " ".join(
                [sentence.replace(" ", "") for sentence in data["article"]]
            )
            summary = data["summary"].replace(" ", "")
            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: article, _ABSTRACT: summary}
            yield id_, raw_feature_info
