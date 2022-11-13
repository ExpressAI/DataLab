from pathlib import Path
from typing import List

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{guntara-etal-2020-benchmarking,
    title = "Benchmarking Multidomain {E}nglish-{I}ndonesian Machine Translation",
    author = "Guntara, Tri Wahyu  and
      Aji, Alham Fikri  and
      Prasojo, Radityo Eko",
    booktitle = "Proceedings of the 13th Workshop on Building and Using Comparable Corpora",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.bucc-1.6",
    pages = "35--43",
    language = "English",
    ISBN = "979-10-95546-42-9",
}
"""

_DESCRIPTION = """\
"In the context of Machine Translation (MT) from-and-to English, Bahasa Indonesia has been considered a low-resource language,
and therefore applying Neural Machine Translation (NMT) which typically requires large training dataset proves to be problematic.
In this paper, we show otherwise by collecting large, publicly-available datasets from the Web, which we split into several domains: news, religion, general, and
conversation,to train and benchmark some variants of transformer-based NMT models across the domains.
We show using BLEU that our models perform well across them , outperform the baseline Statistical Machine Translation (SMT) models,
and perform comparably with Google Translate. Our datasets (with the standard split for training, validation, and testing), code, and models are available on https://github.com/gunnxx/indonesian-mt-data."
"""

_HOMEPAGE = "https://github.com/gunnxx/indonesian-mt-data"

_LICENSE = "Creative Commons Attribution Share-Alike 4.0 International"

_URLS = "https://github.com/gunnxx/indonesian-mt-data/archive/refs/heads/master.zip"


class IndonesianMTConfig(datalabs.BuilderConfig):
    """BuilderConfig for IndonesianMT Config"""

    def __init__(self, **kwargs):
        """BuilderConfig for IndonesianMT Config.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IndonesianMTConfig, self).__init__(**kwargs)


class IndonesianMT(datalabs.GeneratorBasedBuilder):
    """Indonesian General Domain MT En-Id is a machine translation dataset containing English-Indonesian parallel multi-domain sentences."""

    BUILDER_CONFIGS = [
        IndonesianMTConfig(
            name=domain,
            version=datalabs.Version("1.0.0"),
            description=f"Indonesian-English MT for {domain} domain",
        )
        for domain in ["news", "religious"]
    ]

    def _info(self):
        features = datalabs.Features(
            {
                "translation": {
                    "en": datalabs.Value("string"),
                    "id": datalabs.Value("string"),
                },
            }
        )

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # datasets.features.FeatureConnectors
            features=features,
            supervised_keys=None,
            languages=["en", "id"],
            task_templates=[
                get_task(TaskType.machine_translation)(
                    translation_column="translation",
                    lang_sub_columns=["en", "id"],
                )
            ],
        )

    def _split_generators(
        self, dl_manager: datalabs.DownloadManager
    ) -> List[datalabs.SplitGenerator]:
        urls = _URLS
        domain = self.config.name
        data_dir = (
            Path(dl_manager.download_and_extract(urls))
            / "indonesian-mt-data-master"
            / f"{domain}"
        )

        if domain == "general":
            en_train_data_dir = [data_dir / f"train.en.{i}" for i in range(4)]
            id_train_data_dir = [data_dir / f"train.id.{i}" for i in range(4)]
        elif domain == "religious":
            en_train_data_dir = [data_dir / f"train.en.{i}" for i in range(2)]
            id_train_data_dir = [data_dir / f"train.id.{i}" for i in range(2)]
        elif domain == "news":
            en_train_data_dir = [data_dir / f"train.en"]
            id_train_data_dir = [data_dir / f"train.id"]

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "en": en_train_data_dir,
                        "id": id_train_data_dir,
                    }
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": {
                        "en": [data_dir / "test.en"],
                        "id": [data_dir / "test.id"],
                    }
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {
                        "en": [data_dir / "valid.en"],
                        "id": [data_dir / "valid.id"],
                    }
                },
            ),
        ]

    def _generate_examples(self, filepath: dict):
        data_en = None
        for file in filepath["en"]:
            if data_en is None:
                data_en = open(file, "r").readlines()
            else:
                data_en += open(file, "r").readlines()

        data_id = None
        for file in filepath["id"]:
            if data_id is None:
                data_id = open(file, "r").readlines()
            else:
                data_id += open(file, "r").readlines()

        data_en = list(map(str.strip, data_en))
        data_id = list(map(str.strip, data_id))

        for id, (src, tgt) in enumerate(zip(data_en, data_id)):
            yield id, {"translation": {"en": src, "id": tgt}}
