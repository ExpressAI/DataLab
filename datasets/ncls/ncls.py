"""NCLS cross-lingual abstractive summarization dataset."""
import os

import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@inproceedings{zhu-etal-2019-ncls,
    title = "{NCLS}: Neural Cross-Lingual Summarization",
    author = "Zhu, Junnan  and Wang, Qian  and Wang, Yining  and Zhou, Yu and Zhang, Jiajun  and Wang, Shaonan  and Zong, Chengqing",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1302",
    doi = "10.18653/v1/D19-1302",
    pages = "3045--3055",
}
"""

_DESCRIPTION = """\
We propose a novel round-trip translation strategy to acquire large-scale CLS datasets
from existing large-scale MS datasets. We have constructed a 370K English-to-Chinese
(En2Zh) CLS corpus and a 1.69M Chineseto-English (Zh2En) CLS corpus.
see: https://aclanthology.org/D19-1302/
"""

_HOMEPAGE = "https://github.com/ZNLP/NCLS-Corpora"
_LICENSE = "Berkeley Software Distribution license (BSD License)"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class NCLSConfig(datalabs.BuilderConfig):
    """BuilderConfig for NCLS."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for NCLS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NCLSConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class NCLSDataset(datalabs.GeneratorBasedBuilder):
    """NCLS Dataset."""

    _FILE_ID = "1GZpKkHnTH_1Wxiti0BrrxPm18y9rTQRL"
    BUILDER_CONFIGS = [
        NCLSConfig(
            name="en2zh",
            version=datalabs.Version("1.0.0"),
            description="NCLS Dataset for cross-lingual summarization, English-->Chinese version",
            task_templates=[
                Summarization(text_column=_ARTICLE, summary_column=_ABSTRACT)
            ],
        )
    ]
    DEFAULT_CONFIG_NAME = "en2zh"

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
            license=_LICENSE,
            version=self.VERSION,
            languages=["en", "zh"],
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
                    "f_path": os.path.join(
                        f_path,
                        "NCLS-Data/{}SUM".format(str(self.config.name).upper()),
                        "{}SUM".format(str(self.config.name).upper()) + "_train.txt",
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path,
                        "NCLS-Data/{}SUM".format(str(self.config.name).upper()),
                        "{}SUM".format(str(self.config.name).upper()) + "_valid.txt",
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path,
                        "NCLS-Data/{}SUM".format(str(self.config.name).upper()),
                        "{}SUM".format(str(self.config.name).upper()) + "_test.txt",
                    )
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate NCLS examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        summaries = []

        if "_test" in f_path:
            for index, line in enumerate(lines):
                line = line.strip()
                if line == "<Article>":
                    article = lines[index + 1].strip()
                elif "<ZH-REF" in line and "-human-corrected" in line:
                    summaries.append(lines[index + 1].strip())
                elif line == "</doc>":
                    datas.append((article, " ".join(summaries).strip()))
                    summaries = []
        else:
            for index, line in enumerate(lines):
                line = line.strip()
                if line == "<Article>":
                    article = lines[index + 1].strip()
                elif "<ZH-REF" in line:
                    summaries.append(lines[index + 1].strip())
                elif line == "</doc>":
                    datas.append((article, " ".join(summaries).strip()))
                    summaries = []

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: article, _ABSTRACT: summary}
            yield id_, raw_feature_info
