"""QBSUM:  A large-scale query-based document summarization dataset from real-world applications"""
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{DBLP:journals/csl/ZhaoYLZHCNLG21,
  author    = {Mingjun Zhao and
               Shengli Yan and
               Bang Liu and
               Xinwang Zhong and
               Qian Hao and
               Haolan Chen and
               Di Niu and
               Bowei Long and
               Weidong Guo},
  title     = {{QBSUM:} {A} large-scale query-based document summarization dataset
               from real-world applications},
  journal   = {Comput. Speech Lang.},
  volume    = {66},
  pages     = {101166},
  year      = {2021},
  url       = {https://doi.org/10.1016/j.csl.2020.101166},
  doi       = {10.1016/j.csl.2020.101166},
  timestamp = {Thu, 17 Dec 2020 16:21:15 +0100},
  biburl    = {https://dblp.org/rec/journals/csl/ZhaoYLZHCNLG21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
QBSUM is a high-quality large-scale dataset consisting of 49,000+ data samples for the task of Chinese query-based document summarization
see: https://doi.org/10.1016/j.csl.2020.101166
"""

_HOMEPAGE = "https://doi.org/10.1016/j.csl.2020.101166"
_ABSTRACT = "summary"
_ARTICLE = "text"
_KEY = "query"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class QBSUMConfig(datalabs.BuilderConfig):
    """BuilderConfig for QBSUM."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for QBSUM.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QBSUMConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class QBSUMDataset(datalabs.GeneratorBasedBuilder):
    """QBSUM Dataset."""

    _TRAIN_URL = _gdrive_url("1JN70GGAluMdSm9B6PqoI5v7RV6M714FN")
    _VAL_URL = _gdrive_url("18ZiVBSo2VBSiYuiKffjF23dXgIOf-A9v")
    _TEST_URL = _gdrive_url("1S1TEuz-8TyRJ-8x8Xwo0rdpxFktVlJYI")
    BUILDER_CONFIGS = [
        QBSUMConfig(
            name="query-based",
            version=datalabs.Version("1.0.0"),
            description=f"QBSUM Dataset for query-based summarization",
            task_templates=[
                get_task(TaskType.query_summarization)(
                    source_column=_ARTICLE,
                    reference_column=_ABSTRACT,
                    guidance_column=_KEY,
                )
            ],
        )
    ]
    DEFAULT_CONFIG_NAME = "query-based"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    _KEY: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[
                get_task(TaskType.query_summarization)(
                    source_column=_ARTICLE,
                    reference_column=_ABSTRACT,
                    guidance_column=_KEY,
                )
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._TRAIN_URL)
        val_path = dl_manager.download(self._VAL_URL)
        test_path = dl_manager.download(self._TEST_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": val_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_path},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate QBSUM examples."""
        cnt = 0
        buffer = []
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.split("\t")
                if len(line) != 3:
                    continue
                if len(buffer) == 0:
                    buffer.append(line)
                elif buffer[0][0] == line[0]:
                    buffer.append(line)
                else:
                    query = buffer[0][0]
                    article, summary = [], []
                    for x in buffer:
                        article.append(x[1])
                        if x[2] == "Y":
                            summary.append(x[1])
                    yield cnt, {
                        _ARTICLE: " ".join(article),
                        _ABSTRACT: " ".join(summary),
                        _KEY: query,
                    }
                    cnt += 1
                    buffer = [line]
        if len(buffer) > 0:
            query = buffer[0][0]
            article, summary = [], []
            for x in buffer:
                article.append(x[1])
                if x[2] == "Y":
                    summary.append(x[1])
            yield cnt, {
                _ARTICLE: " ".join(article),
                _ABSTRACT: " ".join(summary),
                _KEY: query,
            }
