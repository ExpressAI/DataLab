"""CrossSum: Beyond English-Centric Cross-Lingual Abstractive Text Summarization for 1500+ Language Pairs"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{Hasan2021CrossSumBE,
  title={CrossSum: Beyond English-Centric Cross-Lingual Abstractive Text Summarization for 1500+ Language Pairs},
  author={Tahmid Hasan and Abhik Bhattacharjee and Wasi Uddin Ahmad and Yuan-Fang Li and Yong-Bin Kang and Rifat Shahriyar},
  journal={ArXiv},
  year={2021},
  volume={abs/2112.08804}
}
"""

_DESCRIPTION = """\
We present CrossSum, a large-scale dataset comprising 1.65 million cross-lingual article-summary 
samples in 1500+ language-pairs constituting 45 languages.
see: https://arxiv.org/pdf/2112.08804.pdf
"""

_HOMEPAGE = "https://github.com/csebuetnlp/CrossSum"
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class CrossSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for CrossSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for CrossSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CrossSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class CrossSumDataset(datalabs.GeneratorBasedBuilder):
    """CrossSum Dataset."""

    _FILE_ID = "1HvzrOFPzD2CSvZICUQw4a4EbtX91-hCD"
    _LANG = [
        "am",
        "ar",
        "az",
        "bn",
        "my",
        "zh-CN",
        "zh-TW",
        "en",
        "fr",
        "gu",
        "ha",
        "hi",
        "ig",
        "id",
        "ja",
        "rn",
        "ko",
        "ky",
        "mr",
        "np",
        "om",
        "ps",
        "fa",
        "pcm",
        "pt",
        "pa",
        "ru",
        "gd",
        "sr-C",
        "sr-L",
        "si",
        "so",
        "es",
        "sw",
        "ta",
        "te",
        "th",
        "ti",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "cy",
        "yo",
    ]
    _LANG_NAME = {
        "am": "amharic",
        "ar": "arabic",
        "az": "azerbaijani",
        "bn": "bengali",
        "my": "burmese",
        "zh-CN": "chinese_simplified",
        "zh-TW": "chinese_traditional",
        "en": "english",
        "fr": "french",
        "gu": "gujarati",
        "ha": "hausa",
        "hi": "hindi",
        "ig": "igbo",
        "id": "indonesian",
        "ja": "japanese",
        "rn": "kirundi",
        "ko": "korean",
        "ky": "kyrgyz",
        "mr": "marathi",
        "np": "nepali",
        "om": "oromo",
        "ps": "pashto",
        "fa": "persian",
        "pcm": "pidgin",
        "pt": "portuguese",
        "pa": "punjabi",
        "ru": "russian",
        "gd": "scottish_gaelic",
        "sr-C": "serbian_cyrillic",
        "sr-L": "serbian_latin",
        "si": "sinhala",
        "so": "somali",
        "es": "spanish",
        "sw": "swahili",
        "ta": "tamil",
        "te": "telugu",
        "th": "thai",
        "ti": "tigrinya",
        "tr": "turkish",
        "uk": "ukrainian",
        "ur": "urdu",
        "uz": "uzbek",
        "vi": "vietnamese",
        "cy": "welsh",
        "yo": "yoruba",
    }

    _LANG_PAIRS = []
    for src_lang in _LANG:
        for tgt_lang in _LANG:
            _LANG_PAIRS.append(f"{src_lang}-{tgt_lang}")

    BUILDER_CONFIGS = list(
        CrossSumConfig(
            name=l2l,
            version=datalabs.Version("1.0.0"),
            description=f"CrossSum Dataset for crosslingual summarization, {l2l} split",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        )
        for l2l in _LANG_PAIRS
    )

    DEFAULT_CONFIG_NAME = "en-en"

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
            languages=[self.config.name],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        src_id, tgt_id = self.config.name.split("-")
        src_lang = self._LANG_NAME[src_id]
        tgt_lang = self._LANG_NAME[tgt_id]

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path,
                        "dataset_CrossSum/{}-{}_train.jsonl".format(src_lang, tgt_lang),
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path,
                        "dataset_CrossSum/{}-{}_val.jsonl".format(src_lang, tgt_lang),
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path,
                        "dataset_CrossSum/{}-{}_test.jsonl".format(src_lang, tgt_lang),
                    )
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate CrossSum examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            line = line.strip()
            data = json.loads(line)
            text = data["text"]
            summary = data["summary"]
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
