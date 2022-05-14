# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and DataLab Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os

import datalabs
from datalabs import get_task, TaskType


# TODO(mlqa): BibTeX citation
_CITATION = """\
@article{lewis2019mlqa,
  title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
  author={Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
  journal={arXiv preprint arXiv:1910.07475},
  year={2019}
}
"""

# TODO(mlqa):
_DESCRIPTION = """\
    MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
    MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
    German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
    4 different languages on average.
"""

LANG_URLS = {
    "ar": "https://drive.google.com/uc?export=download&id=1JNCUueSx83puRasdtbxBMHGsHzWXUc98",
    "de": "https://drive.google.com/uc?export=download&id=1Uf-J2SwUnRxqsnc6wXgc2zplACp00gOE",
    "en": "https://drive.google.com/uc?export=download&id=1AkbRfBYC-_L7sk-SNwrTe3aB4cFDbMaY",
    "es": "https://drive.google.com/uc?export=download&id=1V6HlNI66lp_aMtyxRkiBjpuHHE-DG5WA",
    "hi": "https://drive.google.com/uc?export=download&id=1by3t8RAhoZre8_3uOijPhpZ1dMU68xea",
    "vi": "https://drive.google.com/uc?export=download&id=1e-LU8ADBGbG1r8h6Q8XJQsddq4tA9Mvt",
    "zh": "https://drive.google.com/uc?export=download&id=13J-Dn-5ihOizMt-XtARtkvWoTpbSVjMe",
}

class MlqaConfig(datalabs.BuilderConfig):
    def __init__(self, data_url, **kwargs):
        """BuilderConfig for MLQA
        Args:
          data_url: `string`, url to the dataset
          **kwargs: keyword arguments forwarded to super.
        """
        def __init__(self, **kwargs):
            """
            Args:
                **kwargs: keyword arguments forwarded to super.
            """
            super(MlqaConfig, self).__init__(version=datalabs.Version("2.0.0", ""), **kwargs)


class Mlqa(datalabs.GeneratorBasedBuilder):
    """TODO(mlqa): Short description of my dataset."""

    # TODO(mlqa): Set up version.
    VERSION = datalabs.Version("2.0.0")
    BUILDER_CONFIGS = [
        datalabs.BuilderConfig(
            name="{}".format(lang),
            version=datalabs.Version("2.0.0")
        )
        for lang in list(LANG_URLS.keys())
    ]

    def _info(self):
        # TODO(mlqa): Specifies the datasets.DatasetInfo object
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "context": datalabs.Value("string"),
                    "question": datalabs.Value("string"),
                    "answers":
                        {
                            "text": datalabs.features.Sequence(datalabs.Value("string")),
                            "answer_start": datalabs.features.Sequence(datalabs.Value("int32")),
                        }
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/facebookresearch/MLQA",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.qa_extractive)(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(mlqa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        lang = str(self.config.name)
        url = LANG_URLS[lang]
        # url = _URL.format(lang, self.VERSION.version_str[:-2])
        data_dir = dl_manager.download_and_extract(url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "/test-" + lang + ".json"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "/dev-" + lang + ".json"),
                },
            ),
        ]

    def _generate_examples(self, filepath, files=None):
        """Yields examples."""
        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield key, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1
