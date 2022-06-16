import json
import os

import datalabs
from datalabs import get_task, TaskType

_DESCRIPTION = """
 The PENS dataset contains 113,762 pieces of News whose topics are distributed into 15 categories. 
 Each news includes a news ID, a title, a body and a category manually tagged by editors. 
 The average length of news title and news body is 10.5 and 549.0, individually.
 The dataset is casted as a headline generation task, and it only has the training set.
 See: https://aclanthology.org/2021.acl-long.7.pdf
"""
_CITATION = """\
    @inproceedings{ao-etal-2021-pens,
    title = "{PENS}: A Dataset and Generic Framework for Personalized News Headline Generation",
    author = "Ao, Xiang  and
      Wang, Xiting  and
      Luo, Ling  and
      Qiao, Ying  and
      He, Qing  and
      Xie, Xing",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.7",
    doi = "10.18653/v1/2021.acl-long.7",
    pages = "82--92",
    abstract = "In this paper, we formulate the personalized news headline generation problem whose goal is to output a user-specific title based on both a user{'}s reading interests and a candidate news body to be exposed to her. To build up a benchmark for this problem, we publicize a large-scale dataset named PENS (PErsonalized News headlineS). The training set is collected from user impressions logs of Microsoft News, and the test set is manually created by hundreds of native speakers to enable a fair testbed for evaluating models in an offline mode. We propose a generic framework as a preparatory solution to our problem. At its heart, user preference is learned by leveraging the user behavioral data, and three kinds of user preference injections are proposed to personalize a text generator and establish personalized headlines. We investigate our dataset by implementing several state-of-the-art user modeling methods in our framework to demonstrate a benchmark score for the proposed dataset. The dataset is available at https://msnews.github.io/pens.html.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


class PENSConfig(datalabs.BuilderConfig):
    """BuilderConfig for PENS."""

    def __init__(self, **kwargs):
        """BuilderConfig for PENS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PENSConfig, self).__init__(**kwargs)


class PENSDataset(datalabs.GeneratorBasedBuilder):
    """PENS Dataset."""

    BUILDER_CONFIGS = [
        PENSConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="PENS dataset for headline summarization",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):

        features_dataset = {}
        features_sample = datalabs.Features(
            {
                _ARTICLE: datalabs.Value("string"),
                _ABSTRACT: datalabs.Value("string"),
            }
        )

        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage="https://msnews.github.io/pens.html",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(
            "https://msrshare.blob.core.windows.net/msr/training_set.zip"
        )
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "./training_set/news.tsv")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate PENS examples."""
        with open(f_path, encoding="utf-8") as f:
            # read the file using csv dict reader with the delimiter as a tab
            f.readline()
            for id_, x in enumerate(f):
                x = x.strip().split("\t")
                yield id_, {"text": x[4].strip(), "summary": x[3].strip()}
