"""CSDS: Chinese Dataset for Customer Service Dialogue Summarization."""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{lin-etal-2021-csds,
    title = "{CSDS}: A Fine-Grained {C}hinese Dataset for Customer Service Dialogue Summarization",
    author = "Lin, Haitao  and
      Ma, Liqun  and
      Zhu, Junnan  and
      Xiang, Lu  and
      Zhou, Yu  and
      Zhang, Jiajun  and
      Zong, Chengqing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.365",
    pages = "4436--4451",
    abstract = "Dialogue summarization has drawn much attention recently. Especially in the customer service domain, agents could use dialogue summaries to help boost their works by quickly knowing customer{'}s issues and service progress. These applications require summaries to contain the perspective of a single speaker and have a clear topic flow structure, while neither are available in existing datasets. Therefore, in this paper, we introduce a novel Chinese dataset for Customer Service Dialogue Summarization (CSDS). CSDS improves the abstractive summaries in two aspects: (1) In addition to the overall summary for the whole dialogue, role-oriented summaries are also provided to acquire different speakers{'} viewpoints. (2) All the summaries sum up each topic separately, thus containing the topic-level structure of the dialogue. We define tasks in CSDS as generating the overall summary and different role-oriented summaries for a given dialogue. Next, we compare various summarization methods on CSDS, and experiment results show that existing methods are prone to generate redundant and incoherent summaries. Besides, the performance becomes much worse when analyzing the performance on role-oriented summaries and topic structures. We hope that this study could benchmark Chinese dialogue summarization and benefit further studies.",
}
"""

_DESCRIPTION = """\
In this paper, we introduce a novel Chinese dataset for Customer Service Dialogue Summarization (CSDS). CSDS
improves the abstractive summaries in two aspects: (1) In addition to the overall summary
for the whole dialogue, role-oriented summaries are also provided to acquire different
speakers’ viewpoints. (2) All the summaries sum up each topic separately, thus containing the topic-level structure of the dialogue.
see: https://aclanthology.org/2021.emnlp-main.365/
"""

_HOMEPAGE = "https://github.com/xiaolinAndy/CSDS"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"


class CSDSConfig(datalabs.BuilderConfig):
    """BuilderConfig for CSDS."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for CSDS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CSDSConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class CSDSDataset(datalabs.GeneratorBasedBuilder):
    """CSDS Dataset."""

    _FILE_ID = {
        "train": "1-xwmTxDAZXlt3YhPBgQWETzw_yM5Xui2",
        "valid": "1JLW5iRUjdFz1BUGypyGHAm6vEkMI20sS",
        "test": "1xpHJWJd5kLnq9tKqzqE-928FgYEryka1",
    }
    BUILDER_CONFIGS = [
        CSDSConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="CSDS dataset for Chinese customer service summarization, single document version",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        CSDSConfig(
            name="usersumm",
            version=datalabs.Version("1.0.0"),
            description="CSDS dataset for Chinese customer service summarization, dialogue summarization version, target is the user summary.",
            task_templates=[
                get_task(TaskType.dialog_summarization)(
                    source_column="dialogue", reference_column="user_summary"
                )
            ],
        ),
        CSDSConfig(
            name="agentsumm",
            version=datalabs.Version("1.0.0"),
            description="CSDS dataset for Chinese customer service summarization, dialogue summarization version, target is the agent summary.",
            task_templates=[
                get_task(TaskType.dialog_summarization)(
                    source_column="dialogue", reference_column="agent_summary"
                )
            ],
        ),
        CSDSConfig(
            name="finalsumm",
            version=datalabs.Version("1.0.0"),
            description="CSDS dataset for Chinese customer service summarization, dialogue summarization version, target is the final summary.",
            task_templates=[
                get_task(TaskType.dialog_summarization)(
                    source_column="dialogue", reference_column="final_summary"
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):

        if self.config.name == "document":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        elif self.config.name == "usersumm":
            features_sample = datalabs.Features(
                {
                    "dialogue": datalabs.Sequence(
                        datalabs.Features(
                            {
                                "speaker": datalabs.Value("string"),
                                "text": datalabs.Value("string"),
                            }
                        )
                    ),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                }
            )
        elif self.config.name == "agentsumm":
            features_sample = datalabs.Features(
                {
                    "dialogue": datalabs.Sequence(
                        datalabs.Features(
                            {
                                "speaker": datalabs.Value("string"),
                                "text": datalabs.Value("string"),
                            }
                        )
                    ),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                }
            )
        elif self.config.name == "finalsumm":
            features_sample = datalabs.Features(
                {
                    "dialogue": datalabs.Sequence(
                        datalabs.Features(
                            {
                                "speaker": datalabs.Value("string"),
                                "text": datalabs.Value("string"),
                            }
                        )
                    ),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["zh"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        train_f_path = dl_manager.download(_gdrive_url(self._FILE_ID["train"]))
        valid_f_path = dl_manager.download(_gdrive_url(self._FILE_ID["valid"]))
        test_f_path = dl_manager.download(_gdrive_url(self._FILE_ID["test"]))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate CSDS examples."""

        with open(f_path, encoding="utf-8") as f:
            datas = json.load(f)

        for (id_, data) in enumerate(datas):

            if self.config.name == "document":

                article = " ".join(
                    [
                        item["speaker"] + ":" + item["utterance"].replace(" ", "")
                        for item in data["Dialogue"]
                    ]
                )
                summary = " ".join(data["FinalSumm"])

                raw_feature_info = {_ARTICLE: article, _ABSTRACT: summary}

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(
                        raw_feature_info
                    )
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
            else:
                dialogue = []
                for item in data["Dialogue"]:
                    dialogue.append(
                        {
                            "speaker": item["speaker"],
                            "text": item["utterance"].replace(" ", ""),
                        }
                    )

                if self.config.name == "usersumm":
                    summary = data["UserSumm"]
                elif self.config.name == "agentsumm":
                    summary = data["AgentSumm"]
                elif self.config.name == "finalsumm":
                    summary = data["FinalSumm"]

                yield id_, {
                    "dialogue": dialogue,
                    _ABSTRACT: summary,
                }
