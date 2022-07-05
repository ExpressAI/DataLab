"""TWEETSUM: Event-oriented Social Summarization Dataset"""
import os
import csv
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{he-etal-2020-tweetsum,
    title = "{TWEETSUM}: Event oriented Social Summarization Dataset",
    author = "He, Ruifang  and
      Zhao, Liangliang  and
      Liu, Huanyu",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.504",
    doi = "10.18653/v1/2020.coling-main.504",
    pages = "5731--5736"
}
"""

_DESCRIPTION = """\
We construct TWEETSUM, a new event-oriented dataset for social summarization. The original
data is collected from twitter and contains 12 real world hot events with a total of 44,034 tweets
and 11,240 users. We create expert summaries for each event, and we also have the annotation
quality evaluation.
see: https://aclanthology.org/2020.coling-main.504.pdf
"""

_HOMEPAGE = "[official] https://github.com/guyfe/Tweetsumm [processed] https://github.com/sarahaman/CIS6930_TweetSum_Summarization"
_LICENSE = "Creative Commons Zero v1.0 Universal"
_ARTICLE = "text"
_ABSTRACT = "summary"


class TWEETSUMConfig(datalabs.BuilderConfig):
    """BuilderConfig for TWEETSUM."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for TWEETSUM.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TWEETSUMConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class TWEETSUMDataset(datalabs.GeneratorBasedBuilder):
    """TWEETSUM Dataset."""

    BUILDER_CONFIGS = [
        TWEETSUMConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Event-oriented Social Summarization Dataset.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
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
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://github.com/sarahaman/CIS6930_TweetSum_Summarization/archive/refs/heads/main.zip"
        f_path = dl_manager.download_and_extract(url)

        train_f_path = os.path.join(f_path, "CIS6930_TweetSum_Summarization-main/data/tweetsum_train.csv")
        valid_f_path = os.path.join(f_path, "CIS6930_TweetSum_Summarization-main/data/tweetsum_valid.csv")
        test_f_path = os.path.join(f_path, "CIS6930_TweetSum_Summarization-main/data/tweetsum_test.csv")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate TWEETSUM examples."""
        f = open(f_path, encoding="utf-8")
        csvreader = csv.reader(f)
        header = next(csvreader)
        datas = []
        for row in csvreader:
            datas.append((row[0].strip(), row[1].strip()))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
