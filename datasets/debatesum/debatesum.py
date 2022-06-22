"""DebateSum: A large-scale argument mining and summarization dataset"""
import os
import csv

csv.field_size_limit(500 * 1024 * 1024)
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{roush-balaji-2020-debatesum,
    title = "{D}ebate{S}um: A large-scale argument mining and summarization dataset",
    author = "Roush, Allen  and
      Balaji, Arvind",
    booktitle = "Proceedings of the 7th Workshop on Argument Mining",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.argmining-1.1",
    pages = "1--7"
}
"""

_DESCRIPTION = """\
DebateSum consists of 187328 debate documents, arguements (also can be thought of as abstractive summaries, or queries), 
word-level extractive summaries, citations, and associated metadata organized by topic-year. This data is ready for analysis by NLP systems.
see: https://aclanthology.org/2020.argmining-1.1.pdf
"""

_HOMEPAGE = "https://github.com/Hellisotherpeople/DebateSum"
_ARTICLE = "text"
_ABSTRACT = "summary"


class DebateSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for DebateSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for DebateSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DebateSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class DebateSumDataset(datalabs.GeneratorBasedBuilder):
    """DebateSum Dataset."""

    BUILDER_CONFIGS = [
        DebateSumConfig(
            name="extract",
            version=datalabs.Version("1.0.0"),
            description="Argument mining and summarization dataset, word-level extractive summary version.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        DebateSumConfig(
            name="abstract",
            version=datalabs.Version("1.0.0"),
            description="Argument mining and summarization dataset, abstractive summary version.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "extract"

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
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://huggingface.co/datasets/Hellisotherpeople/DebateSum/resolve/main/debateallcsv.csv"
        f_path = dl_manager.download(url)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": f_path}
            )
        ]

    def _generate_examples(self, f_path):
        """Generate TWEETSUM examples."""
        f = open(f_path, encoding="utf-8")
        csvreader = csv.reader(f)
        header = next(csvreader)
        datas = []
        for row in csvreader:
            if self.config.name == "extract":
                datas.append((row[0].strip(), row[2].strip()))
            elif self.config.name == "abstract":
                datas.append((row[0].strip(), row[3].strip()))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
