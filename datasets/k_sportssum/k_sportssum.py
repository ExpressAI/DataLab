"""K-SportsSum: Knowledge Enhanced Sports Game Summarization"""
import os
import json
import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@inproceedings{wang2022knowledge,
  title={Knowledge Enhanced Sports Game Summarization},
  author={Wang, Jiaan and Li, Zhixu and Zhang, Tingyi and Zheng, Duo and Qu, Jianfeng and Liu, An and Zhao, Lei and Chen, Zhigang},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={1045--1053},
  year={2022}
}
"""

_DESCRIPTION = """\
K-SportsSum, it has 7854 sports game summarization samples together with 
a large-scale knowledge corpus containing information of 523 sports teams and 14k+ sports players.
see: https://arxiv.org/pdf/2111.12535.pdf
"""

_HOMEPAGE = "https://github.com/krystalan/K-SportsSum"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class KSportsSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for K-SportsSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for K-SportsSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(KSportsSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class KSportsSumDataset(datalabs.GeneratorBasedBuilder):
    """K-SportsSum Dataset."""
    _FILE_ID = "1RGWIz3Nw_kzfgIYo_Ke9elLfPOg0rS4V"

    BUILDER_CONFIGS = [
        KSportsSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Dataset for sports game summarization.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: {
                        "commentary": datalabs.Sequence(datalabs.Value("string")),
                        "score": datalabs.Sequence(datalabs.Value("string"))
                    },
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=["zh"],
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "train.json")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "val.json")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "test.json")}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate K-SportsSum examples."""
        with open("K-SportsSum/val.json", encoding="utf-8") as f:
            original_datas = json.load(f)

        datas = []
        for original_data in original_datas:
            seqs = original_data["commentary"]
            commentary = [seq[0].strip() + " " + seq[1].strip() for seq in seqs]
            score = [seq[2].strip() for seq in seqs]
            summary = original_data["news"].strip()
            datas.append({"commentary": commentary, "score": score, "summary": summary})

        for id_, data in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: {
                    "commentary": data["commentary"],
                    "score": data["score"]
                },
                _ABSTRACT: data["summary"]
            }
            yield id_, raw_feature_info
