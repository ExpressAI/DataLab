import json
import os
import datalabs
from datalabs import get_task, TaskType




_DESCRIPTION = """
 SportsSum is a sports game summarization dataset in Chinese. The goal of SportsSum is to generate sports summaries from live commentaries.
 See: https://aclanthology.org/2020.aacl-main.61.pdf
"""

_CITATION = """\
    @inproceedings{Huang2020sportssum,
    author    = {Kuan-Hao Huang and
                 Chen Li and
                 Kai-Wei Chang},
    title     = {Generating Sports News from Live Commentary: A Chinese Dataset for Sports Game Summarization},
    booktitle = {Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL)},
    year      = {2020},
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


class SportsSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for SportsSum."""

    def __init__(self, **kwargs):
        """BuilderConfig for SportsSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SportsSumConfig, self).__init__(**kwargs)


class SportsSumDataset(datalabs.GeneratorBasedBuilder):
    """SportsSum Dataset."""
    _FILE_URL = "https://github.com/ej0cl6/SportsSum/blob/master/sports_data.tar.gz?raw=true"
    BUILDER_CONFIGS = [
        SportsSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="SportsSum dataset for summarization, document, containining only the train split",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):


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
            supervised_keys=None,
            homepage="https://github.com/ej0cl6/SportsSum",
            citation=_CITATION,
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._FILE_URL)
        f_path = os.path.join(f_path, "./sports_data")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": f_path, "split": "train"},
            ),
        ]

    def _generate_examples(self, f_path, split):
        """Generate SportsSum examples."""
        cnt = 0
        fdirs = os.listdir(f_path)
        for fdir in fdirs:
            with open(os.path.join(f_path, fdir, "news.txt"), "r") as f:
                summary = f.read().strip()
            with open(os.path.join(f_path, fdir, "live.json"), "r") as f:
                live = json.load(f)
            article = " ".join([x["m"].strip() for x in live["result"]["data"]])
            raw_feature_info = {
                _ARTICLE: article,
                _ABSTRACT: summary,
            }
            yield cnt, raw_feature_info
            cnt += 1


