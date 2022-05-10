# coding=utf-8
# Copyright 2020 The TensorFlow datalab Authors and the HuggingFace datasets, DataLab Authors.
# Usage of dataset is restricted to non-commercial research purposes only. Copyright belongs to the original copyright holders.

"""MLSum: multiligual summarization dataset"""
import os
import datalabs
from datalabs.tasks import Summarization
import json

_CITATION = """\
@inproceedings{scialom-etal-2020-mlsum,
    title = "{MLSUM}: The Multilingual Summarization Corpus",
    author = "Scialom, Thomas  and
      Dray, Paul-Alexis  and
      Lamprier, Sylvain  and
      Piwowarski, Benjamin  and
      Staiano, Jacopo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.647",
    doi = "10.18653/v1/2020.emnlp-main.647",
    pages = "8051--8067",
    abstract = "We present MLSUM, the first large-scale MultiLingual SUMmarization dataset. Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages {--} namely, French, German, Spanish, Russian, Turkish. Together with English news articles from the popular CNN/Daily mail dataset, the collected data form a large scale multilingual dataset which can enable new research directions for the text summarization community. We report cross-lingual comparative analyses based on state-of-the-art systems. These highlight existing biases which motivate the use of a multi-lingual dataset.",
}
"""

_DESCRIPTION = """\
MLSUM, the first large-scale MultiLingual SUMmarization dataset. 
Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages - namely, French, German, Spanish, Russian, Turkish. 
Together with English news articles from the popular CNN/Daily mail dataset, the collected data form a large scale multilingual dataset which can enable new research directions for the text summarization community.
see: https://aclanthology.org/2020.emnlp-main.647.pdf
"""

_HOMEPAGE = "https://github.com/ThomasScialom/MLSUM"
_ABSTRACT = "summary"
_ARTICLE = "text"




class MLSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for MLSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MLSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MLSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MLSumDataset(datalabs.GeneratorBasedBuilder):
    """MLSum Dataset."""
    _LANG = ["de", "es", "fr", "ru", "tu"]
    _URL = "https://gitlab.lip6.fr/scialom/mlsum_data/-/raw/master/MLSUM/"
    BUILDER_CONFIGS = list([
        MLSumConfig(
            name=l,
            version=datalabs.Version("1.0.0"),
            description=f"MLSum Dataset for multiligual summarization, {l} split",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ) for l in ["de", "es", "fr", "ru", "tu"]
    ])
    DEFAULT_CONFIG_NAME = "de"

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
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        lang_id = self.config.name
        train_path = dl_manager.download_and_extract(f"{self._URL}{lang_id}_train.zip")
        test_path = dl_manager.download_and_extract(f"{self._URL}{lang_id}_test.zip")
        val_path = dl_manager.download_and_extract(f"{self._URL}{lang_id}_val.zip")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(train_path, f"{lang_id}_train.jsonl"),
                    }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(val_path, f"{lang_id}_val.jsonl"),
                    }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(test_path, f"{lang_id}_test.jsonl"),
                    }
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate MLSum examples."""
        with open(f_path, encoding="utf-8") as f: 
            for (id_, x) in enumerate(f):
                x = json.loads(x)
                yield id_, {_ARTICLE: x["text"].strip(), _ABSTRACT: x["summary"].strip()}
                