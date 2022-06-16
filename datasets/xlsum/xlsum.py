# coding=utf-8
# Copyright 2022 The HuggingFace datasets Authors, DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""XL-Sum abstractive summarization dataset."""
import json
import os

import datalabs
from datalabs import get_task, TaskType

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
_URL = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/{}_XLSum_v{}.tar.bz2"
_LANGUAGES = {
    "oromo": "or",
    "french": "fr",
    "amharic": "am",
    "arabic": "ar",
    "azerbaijani": "az",
    "bengali": "be",
    "burmese": "my",
    "chinese_simplified": "zh",
    "chinese_traditional": "zh-CHT",
    "welsh": "cy",
    "english": "en",
    "kirundi": "ki",
    "gujarati": "gu",
    "hausa": "ha",
    "hindi": "hi",
    "igbo": "ig",
    "indonesian": "in",
    "japanese": "ja",
    "korean": "ko",
    "kyrgyz": "ky",
    "marathi": "ma",
    "spanish": "es",
    "scottish_gaelic": "gd",
    "nepali": "ne",
    "pashto": "ps",
    "persian": "fa",
    "pidgin": "pcm",
    "portuguese": "pt",
    "punjabi": "pa",
    "russian": "ru",
    "serbian_cyrillic": "sr-Cyrl",
    "serbian_latin": "sr-Latn",
    "sinhala": "si",
    "somali": "so",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "tigrinya": "ti",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yoruba": "yo",
}


class Xlsum(datalabs.GeneratorBasedBuilder):
    VERSION = datalabs.Version("2.0.0")

    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(lang), version=datalabs.Version("2.0.0")
        )
        for lang in _LANGUAGES.keys()
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
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column="text", reference_column="summary"
                )
            ],
            languages=[_LANGUAGES[self.config.name]],
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
