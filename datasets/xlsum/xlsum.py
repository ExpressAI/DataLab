"""XL-Sum abstractive summarization dataset."""
import json
import os
import datalabs
from datalabs.tasks import Summarization


_CITATION = """\
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
"""
_DESCRIPTION = """\
We present XLSum, a comprehensive and diverse dataset comprising 1.35 million professionally 
annotated article-summary pairs from BBC, extracted using a set of carefully designed heuristics.
The dataset covers 45 languages ranging from low to high-resource, for many of which no
public dataset is currently available. XL-Sum is highly abstractive, concise, 
and of high quality, as indicated by human and intrinsic evaluation. 
"""
_HOMEPAGE = "https://github.com/csebuetnlp/xl-sum"
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"
_URL = "https://huggingface.co/datalab/csebuetnlp/xlsum/resolve/main/data/{}_XLSum_v{}.tar.bz2"
_LANGUAGES = [
    "oromo",
    "french",
    "amharic",
    "arabic",
    "azerbaijani",
    "bengali",
    "burmese",
    "chinese_simplified",
    "chinese_traditional",
    "welsh",
    "english",
    "kirundi",
    "gujarati",
    "hausa",
    "hindi",
    "igbo",
    "indonesian",
    "japanese",
    "korean",
    "kyrgyz",
    "marathi",
    "spanish",
    "scottish_gaelic",
    "nepali",
    "pashto",
    "persian",
    "pidgin",
    "portuguese",
    "punjabi",
    "russian",
    "serbian_cyrillic",
    "serbian_latin",
    "sinhala",
    "somali",
    "swahili",
    "tamil",
    "telugu",
    "thai",
    "tigrinya",
    "turkish",
    "ukrainian",
    "urdu",
    "uzbek",
    "vietnamese",
    "yoruba",
]


class Xlsum(datalabs.GeneratorBasedBuilder):
    VERSION = datalabs.Version("2.0.0")

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(lang),
            version=datalabs.Version("2.0.0")
        )
        for lang in _LANGUAGES
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "url": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "summary": datalabs.Value("string"),
                    "text": datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=self.VERSION,
            task_templates=[Summarization(
                text_column="text",
                summary_column="summary"),
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang = str(self.config.name)
        url = _URL.format(lang, self.VERSION.version_str[:-2])
        data_dir = dl_manager.download_and_extract(url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_train.jsonl"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_test.jsonl"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_val.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            for idx_, row in enumerate(f):
                data = json.loads(row)
                yield idx_, {
                    "id": data["id"],
                    "url": data["url"],
                    "title": data["title"],
                    "summary": data["summary"],
                    "text": data["text"],
                }