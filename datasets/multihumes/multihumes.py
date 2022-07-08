"""MultiHumES: Multilingual Humanitarian Response Dataset for Extractive Summarization"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{yela-bello-etal-2021-multihumes,
    title = "{M}ulti{H}um{ES}: Multilingual Humanitarian Dataset for Extractive Summarization",
    author = "Yela-Bello, Jenny Paola  and
      Oglethorpe, Ewan  and
      Rekabsaz, Navid",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.146",
    doi = "10.18653/v1/2021.eacl-main.146",
    pages = "1713--1717",
    abstract = "When responding to a disaster, humanitarian experts must rapidly process large amounts of secondary data sources to derive situational awareness and guide decision-making. While these documents contain valuable information, manually processing them is extremely time-consuming when an expedient response is necessary. To improve this process, effective summarization models are a valuable tool for humanitarian response experts as they provide digestible overviews of essential information in secondary data. This paper focuses on extractive summarization for the humanitarian response domain and describes and makes public a new multilingual data collection for this purpose. The collection {--} called MultiHumES{--} provides multilingual documents coupled with informative snippets that have been annotated by humanitarian analysts over the past four years. We report the performance results of a recent neural networks-based summarization model together with other baselines. We hope that the released data collection can further grow the research on multilingual extractive summarization in the humanitarian response domain.",
}
"""

_DESCRIPTION = """\
MultiHumES provides multilingual documents coupled with informative snippets that have been annotated by humanitarian analysts over the past 
The dataset consists of approximately 50K documents in three languages: English, French, and Spanish. 
Among these documents, approximately 35K are annotated with informative snippets and can be used for the training and evaluation of extractive summarization models
see: https://aclanthology.org/2021.eacl-main.146.pdf
"""

_HOMEPAGE = "https://deephelp.zendesk.com/hc/en-us/articles/360055330172-Overview"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class MultiHumESConfig(datalabs.BuilderConfig):
    """BuilderConfig for MultiHumES."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MultiHumES.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiHumESConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MultiHumESDataset(datalabs.GeneratorBasedBuilder):
    """MultiHumES Dataset."""

    _LANG = ["en", "es", "fr", "multilingual"]
    _FILE_ID = "1ALyhms8XWxuM-w56D9dk1MMMUpDvlGrI"
    BUILDER_CONFIGS = list(
        [
            MultiHumESConfig(
                name=l,
                version=datalabs.Version("1.0.0"),
                description=f"MultiHumES Dataset for multiligual summarization, {l} split",
                task_templates=[
                    get_task(TaskType.summarization)(
                        source_column=_ARTICLE, reference_column=_ABSTRACT
                    )
                ],
            )
            for l in ["en", "es", "fr", "multilingual"]
        ]
    )
    DEFAULT_CONFIG_NAME = "multilingual"

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
        lang_id = self.config.name
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))
        f_path = os.path.join(f_path, "MultiHumES", "docs_with_summaries_formatted")
        if lang_id == "multilingual":
            lang_id = "multi"
        train_src_path = os.path.join(f_path, lang_id, f"{lang_id}.train.src.txt")
        train_tgt_path = os.path.join(f_path, lang_id, f"{lang_id}.train.tgt.txt")
        val_src_path = os.path.join(f_path, lang_id, f"{lang_id}.val.src.txt")
        val_tgt_path = os.path.join(f_path, lang_id, f"{lang_id}.val.tgt.txt")
        test_src_path = os.path.join(f_path, lang_id, f"{lang_id}.test.src.txt")
        test_tgt_path = os.path.join(f_path, lang_id, f"{lang_id}.test.tgt.txt")
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
        """Generate MultiHumES examples."""
        with open(src_path, encoding="utf-8") as f_src, open(
            tgt_path, encoding="utf-8"
        ) as f_tgt:
            for (id_, (x, y)) in enumerate(zip(f_src, f_tgt)):
                x = x.strip().replace("##SENT##", "")
                y = y.strip().replace("##SENT##", "")
                yield id_, {_ARTICLE: x, _ABSTRACT: y}
