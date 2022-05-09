# coding=utf-8
# Copyright 2020 DataLab Authors.
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

# Lint as: python3
"""MasakhaNER: Named Entity Recognition for African Languages"""

import datalabs
import os

from datalabs.tasks import SequenceLabeling
from datalabs.task_dataset import SequenceLabelingDataset

logger = datalabs.logging.get_logger(__name__)


_CITATION = """\
@article{adelani-etal-2021-masakhaner,
    title = "{M}asakha{NER}: Named Entity Recognition for {A}frican Languages",
    author = "Adelani, David Ifeoluwa and Abbott, Jade and Neubig, Graham and D{'}souza, Daniel and Kreutzer, Julia and Lignos, Constantine and Palen-Michel, Chester and Buzaaba, Happy and Rijhwani, Shruti and Ruder, Sebastian and Mayhew, Stephen and Azime, Israel Abebe and Muhammad, Shamsuddeen H. and Emezue, Chris Chinenye and Nakatumba-Nabende, Joyce and Ogayo, Perez and Anuoluwapo, Aremu and Gitau, Catherine and Mbaye, Derguene and Alabi, Jesujoba and Yimam, Seid Muhie and Gwadabe, Tajuddeen Rabiu and Ezeani, Ignatius and Niyongabo, Rubungo Andre and Mukiibi, Jonathan and Otiende, Verrah and Orife, Iroro and David, Davis and Ngom, Samba and Adewumi, Tosin and Rayson, Paul and Adeyemi, Mofetoluwa and Muriuki, Gerald and Anebi, Emmanuel and Chukwuneke, Chiamaka and Odu, Nkiruka and Wairagala, Eric Peter and Oyerinde, Samuel and Siro, Clemencia and Bateesa, Tobius Saul and Oloyede, Temilola and Wambui, Yvonne and Akinode, Victor and Nabagereka, Deborah and Katusiime, Maurice and Awokoya, Ayodele and MBOUP, Mouhamadane and Gebreyohannes, Dibora and Tilaye, Henok and Nwaike, Kelechi and Wolde, Degaga and Faye, Abdoulaye and Sibanda, Blessing and Ahia, Orevaoghene and Dossou, Bonaventure F. P. and Ogueji, Kelechi and DIOP, Thierno Ibrahima and Diallo, Abdoulaye and Akinfaderin, Adewale and Marengereke, Tendai and Osei, Salomey",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.66",
    doi = "10.1162/tacl_a_00416",
    pages = "1116--1131",
}
"""

_DESCRIPTION = """\
MasakhaNER is a named entity recognition corpus that takes a step towards addressing
the under-representation of the African continent in NLP research by creating the first 
large publicly available high-quality dataset for named entity recognition (NER) in 
African languages. This repository corresponds to version 2.0.
"""

_SUPPORTED_LANGUAGES = [
    'bbj',
    'ewe',
    'fon',
    'hau',
    'ibo',
    'kin',
    'lug',
    'mos',
    'nya',
    'pcm',
    'sna',
    'swa',
    'tsn',
    'twi',
    'wol',
    'xho',
    'yor',
    'zul',
]
_LABEL_CLASSES = [
    "O",
    "B-DATE",
    "I-DATE",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-PER",
    "I-PER",
]
_BASE_URL = "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/MasakhaNER2.0/data"


class MasakhaNERConfig(datalabs.BuilderConfig):
    """BuilderConfig for MasakhaNER"""

    def __init__(self, language, **kwargs):
        """
        Args:
            language: the language to consider (in iso 639-3 language code)
            **kwargs: keyword arguments forwarded to super.
        """
        self.language = language
        super().__init__(**kwargs)


class MasakhaNER(datalabs.GeneratorBasedBuilder):
    """MasakhaNER dataset."""

    BUILDER_CONFIG_CLASS = MasakhaNERConfig
    BUILDER_CONFIGS = [
        MasakhaNERConfig(
            name=f"masakhaner-{language}",
            description=f"MasakhaNER language for {language}",
            version=datalabs.Version("1.0.0"),
            language=language,
        )
        for language in _SUPPORTED_LANGUAGES
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "tokens": datalabs.Sequence(datalabs.Value("string")),
                    "tags": datalabs.Sequence(
                        datalabs.features.ClassLabel(names=_LABEL_CLASSES)
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
            task_templates=[SequenceLabeling(tokens_column="tokens", tags_column="tags",
                                                          task="named-entity-recognition")]
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang_url = f'{_BASE_URL}/{self.config.language}'
        train_file = dl_manager.download(f'{lang_url}/train.txt')
        dev_file = dl_manager.download(f'{lang_url}/dev.txt')
        test_file = dl_manager.download(f'{lang_url}/test.txt')

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_file}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": dev_file}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s\n", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:

                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "tags": tags,
                        }
                        guid += 1
                        tokens = []
                        tags = []
                else:
                    # MasakhaNER tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    tags.append(splits[1])

            # last example
            if len(tokens) != 0:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "tags": tags,
                }
