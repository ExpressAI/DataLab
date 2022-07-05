import json
import os
import datalabs
from datalabs import get_task, TaskType




_DESCRIPTION = """
 SportsSum2.0 is a Chinese sports game summarization dataset which is based on SportsSum. In short, SportsSum2.0 is the cleaned version of SportsSum. Sports Game Summarization is a challenging task, which aims to generate sports summaries (i.e., news articles) from corresponding live commentaries.
"""

_CITATION = """\
    @article{Wang2021SportsSum20GH,
  title={SportsSum2.0: Generating High-Quality Sports News from Live Text Commentary},
  author={Jiaan Wang and Zhixu Li and Qiang Yang and Jianfeng Qu and Zhigang Chen and Qingsheng Liu and Guoping Hu},
  journal={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  year={2021}
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"

class SportsSumV2Config(datalabs.BuilderConfig):
    """BuilderConfig for SportsSumV2."""

    def __init__(self, **kwargs):
        """BuilderConfig for SportsSumV2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SportsSumV2Config, self).__init__(**kwargs)


class SportsSumV2Dataset(datalabs.GeneratorBasedBuilder):
    """SportsSumV2 Dataset."""
    BUILDER_CONFIGS = [
        SportsSumV2Config(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="SportsSumV2 dataset for summarization, document, containining only the train split",
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
            homepage="https://github.com/krystalan/SportsSum2.0",
            citation=_CITATION,
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url("1NnXkMqBb1BUq7WMN06t8vZqh8NrD1XZ8"))
        f_path = os.path.join(f_path, "./SportsSum2.0/")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": f_path},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate SportsSumV2 examples."""
        cnt = 0
        fdirs = os.listdir(f_path)
        for fdir in fdirs:
            with open(os.path.join(f_path, fdir, f"{fdir}.txt"), "r") as f:
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


