"""ConvoSumm: Conversation Summarization Benchmark"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{fabbri-etal-2021-convosumm,
    title = "{C}onvo{S}umm: Conversation Summarization Benchmark and Improved Abstractive Summarization with Argument Mining",
    author = "Fabbri, Alexander  and
      Rahman, Faiaz  and
      Rizvi, Imad  and
      Wang, Borui  and
      Li, Haoran  and
      Mehdad, Yashar  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.535",
    doi = "10.18653/v1/2021.acl-long.535",
    pages = "6866--6880",
    abstract = "While online conversations can cover a vast amount of information in many different formats, abstractive text summarization has primarily focused on modeling solely news articles. This research gap is due, in part, to the lack of standardized datasets for summarizing online discussions. To address this gap, we design annotation protocols motivated by an issues{--}viewpoints{--}assertions framework to crowdsource four new datasets on diverse online conversation forms of news comments, discussion forums, community question answering forums, and email threads. We benchmark state-of-the-art models on our datasets and analyze characteristics associated with the data. To create a comprehensive benchmark, we also evaluate these models on widely-used conversation summarization datasets to establish strong baselines in this domain. Furthermore, we incorporate argument mining through graph construction to directly model the issues, viewpoints, and assertions present in a conversation and filter noisy input, showing comparable or improved results according to automatic and human evaluations.",
}
"""

_DESCRIPTION = """\
ConvoSumm contains four new datasets on diverse online conversation forms of news comments, discussion forums, community question answering forums, and email threads.
Several key categories of data are identified for which standard human-created developmentand testing datasets do not exist, namely (1) news article comments, (2) discussion forums and debate,
(3) community question answering, and (4) email threads. 
see: https://aclanthology.org/2021.acl-long.535.pdf
"""

_HOMEPAGE = "https://github.com/Yale-LILY/ConvoSumm"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"


class ConvoSummConfig(datalabs.BuilderConfig):
    """BuilderConfig for ConvoSumm."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for ConvoSumm.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ConvoSummConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class ConvoSummDataset(datalabs.GeneratorBasedBuilder):
    """ConvoSumm Dataset."""

    BUILDER_CONFIGS = list(
        [
            ConvoSummConfig(
                name=l,
                version=datalabs.Version("1.0.0"),
                description=f"ConvoSumm Dataset for conversation summarization, {l} split",
                task_templates=[
                    get_task(TaskType.summarization)(
                        source_column=_ARTICLE, reference_column=_ABSTRACT
                    )
                ],
            )
            for l in ["NYT", "Reddit", "Stack", "Email"]
        ]
    )
    DEFAULT_CONFIG_NAME = "NYT"

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
            languages=[self.config.name],
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "NYT":
            train_src_path = dl_manager.download(
                _gdrive_url("1lmf93mJyoJEai79Awvw9sdcD8yYY9WbP")
            )
            train_tgt_path = dl_manager.download(
                _gdrive_url("1z6-F2ulKvVJtPwzRGYZ8V2nqKnxHOWgS")
            )
            val_src_path = dl_manager.download(
                _gdrive_url("1QUFseVr0ehnAyj2FktoNKMJ8nu1AOupM")
            )
            val_tgt_path = dl_manager.download(
                _gdrive_url("1PVhY6RayffrcA67hLc91XenYRMFDCfCT")
            )
            test_src_path = dl_manager.download(
                _gdrive_url("1e9ajy0jaimH9DeeUG4ZppiPetXf6iuvF")
            )
            test_tgt_path = dl_manager.download(
                _gdrive_url("1kDOK82gpfkkbW3TdpVwQiLm9-JfM7Gnb")
            )
        elif self.config.name == "Reddit":
            train_src_path = dl_manager.download(
                _gdrive_url("1ar08p3ZAzkfnCuvdh2RjyHken5-EbZQ2")
            )
            train_tgt_path = dl_manager.download(
                _gdrive_url("1Bw0euyPZbi5g0q3ft7aVQiVg4j2TijTl")
            )
            val_src_path = dl_manager.download(
                _gdrive_url("1r4SdQr5ioCZGCcha-0lVET2WKobTJWCz")
            )
            val_tgt_path = dl_manager.download(
                _gdrive_url("1bsKbC_XtthgtZfP4MRc-9W4rsojk7_Jm")
            )
            test_src_path = dl_manager.download(
                _gdrive_url("1dywfhJPEaiVn5yZguq7eJsRuAdPYPITj")
            )
            test_tgt_path = dl_manager.download(
                _gdrive_url("1obZ6q7sec9YTx5qub1dbdOZoRWMXBSNR")
            )
        elif self.config.name == "Stack":
            train_src_path = dl_manager.download(
                _gdrive_url("1TOF4xpjmU8PgwD9BxeNn89JMUn9CIoIE")
            )
            train_tgt_path = dl_manager.download(
                _gdrive_url("1h4e6_zku9dfTyZ6eXmHL5f48eFuQN4Ng")
            )
            val_src_path = dl_manager.download(
                _gdrive_url("1cqbLLmJTJGxBnbkjlENaF6yeNUsfiDmE")
            )
            val_tgt_path = dl_manager.download(
                _gdrive_url("1v_OCWo-ZwCMhC_xwEIcJc0oH9J_Mr8SD")
            )
            test_src_path = dl_manager.download(
                _gdrive_url("1JIHO0f0a7XlMPjlaVBgfNt95FNeuDDzH")
            )
            test_tgt_path = dl_manager.download(
                _gdrive_url("1gg9NFWyOgjnE6ql8mOHlJGHohHddWkCl")
            )
        else:
            train_src_path = dl_manager.download(
                _gdrive_url("1ObO0w8FLuusEVCMBAhYl4Fsr-Tud1kmx")
            )
            train_tgt_path = dl_manager.download(
                _gdrive_url("1AcFELfTmpMhkPmt7snv4Xbu_35E7-QIH")
            )
            val_src_path = dl_manager.download(
                _gdrive_url("16LF7UhYZNtgAGZtFEcY_Lnm8whdpkUgY")
            )
            val_tgt_path = dl_manager.download(
                _gdrive_url("163dHY3S8uffpf9qin1aWETIVJ6hBusfn")
            )
            test_src_path = dl_manager.download(
                _gdrive_url("1Hrb5XxRJZbsl320XCBlhU55pOF-fgmAV")
            )
            test_tgt_path = dl_manager.download(
                _gdrive_url("15KsaU_E6NiBD-iyOFgw3PwyBAMXNEASS")
            )
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "src_path": train_src_path,
                    "tgt_path": train_tgt_path,
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "src_path": val_src_path,
                    "tgt_path": val_tgt_path,
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "src_path": test_src_path,
                    "tgt_path": test_tgt_path,
                },
            ),
        ]

    def _generate_examples(self, src_path, tgt_path):
        """Generate ConvoSumm examples."""
        with open(src_path, encoding="utf-8") as f_src, open(
            tgt_path, encoding="utf-8"
        ) as f_tgt:
            for (id_, (x, y)) in enumerate(zip(f_src, f_tgt)):
                yield id_, {_ARTICLE: x.strip(), _ABSTRACT: y.strip()}
