"""WikiLingua: A Multilingual Abstractive Summarization Dataset"""
import os
import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@inproceedings{ladhak-etal-2020-wikilingua,
    title = "{W}iki{L}ingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization",
    author = "Ladhak, Faisal  and
      Durmus, Esin  and
      Cardie, Claire  and
      McKeown, Kathleen",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.360",
    doi = "10.18653/v1/2020.findings-emnlp.360",
    pages = "4034--4048",
    abstract = "We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems. We extract article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard article-summary alignments across languages by aligning the images that are used to describe each how-to step in an article. As a set of baselines for further studies, we evaluate the performance of existing cross-lingual abstractive summarization methods on our dataset. We further propose a method for direct cross-lingual summarization (i.e., without requiring translation at inference time) by leveraging synthetic data and Neural Machine Translation as a pre-training step. Our method significantly outperforms the baseline approaches, while being more cost efficient during inference.",
}
"""

_DESCRIPTION = """\
WikiLingua is a large-scale, multilingual dataset for the evaluation of crosslingual abstractive summarization systems.
The article and summary pairs in 18 languages are extracted from WikiHow12, a high quality,collaborative resource of how-to guides on a diverse set of topics written by human authors.
The gold-standard article-summary alignments across languages are created by aligning the images that are used to describe each how-to step in an article.
see: https://aclanthology.org/2020.findings-emnlp.360
"""

_HOMEPAGE = "https://github.com/esdurmus/Wikilingua"
_LICENSE = "CC BY-NC-SA 3.0"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class WikiLinguaConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiLingua."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WikiLingua.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiLinguaConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WikiLinguaDataset(datalabs.GeneratorBasedBuilder):
    """WikiLingua Dataset."""
    _FILE_ID = "1PM7GFCy2gJL1WHqQz1dzqIDIEN6kfRoi"
    _LANG = ["en", "es", "pt", "fr", "de", "ru", "it", "id", "nl", "ar", "zh", "vi", "th", "ja", "ko", "hi", "cs", "tr"]
    _LANG_NAME = {
        "en": "english",
        "es": "spanish",
        "pt": "portuguese",
        "fr": "french",
        "de": "german",
        "ru": "russian",
        "it": "italian",
        "id": "indonesian",
        "nl": "dutch",
        "ar": "arabic",
        "zh": "chinese",
        "vi": "vietnamese",
        "th": "thai",
        "ja": "japanese",
        "ko": "korean",
        "hi": "hindi",
        "cs": "czech",
        "tr": "turkish",
    }
    BUILDER_CONFIGS = list([
        WikiLinguaConfig(
            name=l,
            version=datalabs.Version("1.0.0"),
            description=f"WikiLingua Dataset for multilingual summarization, {l} split",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ) for l in ["en", "es", "pt", "fr", "de", "ru", "it", "id", "nl", "ar", "zh", "vi", "th", "ja", "ko", "hi", "cs", "tr"]
    ] + [
        WikiLinguaConfig(
            name=f"{l}-en",
            version=datalabs.Version("1.0.0"),
            description=f"WikiLingua Dataset for crosslingual summarization, {l}-en split",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ) for l in ["es", "pt", "fr", "de", "ru", "it", "id", "nl", "ar", "zh", "vi", "th", "ja", "ko", "hi", "cs", "tr"]
    ])
    DEFAULT_CONFIG_NAME = "en"

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
            languages=self.config.name.split('-'),
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))
        if "-" in self.config.name:
            # cross-lingual summarization
            src_id, tgt_id = self.config.name.split("-")
            src_name = self._LANG_NAME[src_id]
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/train.src.{src_id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/train.tgt.{tgt_id}")
                        }
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/val.src.{src_id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/val.tgt.{tgt_id}")
                        }
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/test.src.{src_id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{src_name}/test.tgt.{tgt_id}")
                        }
                ),
            ]
        else:  # multilingual summarization
            id = self.config.name
            name = self._LANG_NAME[id]
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/train.src.{id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/train.tgt.{id}")
                        }
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/val.src.{id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/val.tgt.{id}")
                        }
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    gen_kwargs={
                        "text_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/test.src.{id}"), 
                        "summary_path": os.path.join(f_path, f"./WikiLingua_data_splits/{name}/test.tgt.{id}")
                        }
                ),
            ]

    def _generate_examples(self, text_path, summary_path):
        """Generate WikiLingua examples."""
        with open(text_path, encoding="utf-8") as f_src, open(summary_path, encoding="utf-8") as f_tgt: 
            for (id_, (x, y)) in enumerate(zip(f_src, f_tgt)):
                x = x.strip()
                y = y.strip()
                yield id_, {_ARTICLE: x, _ABSTRACT: y}
                