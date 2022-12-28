import json
import csv
import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.logging import get_logger
import os

logger = get_logger(__name__)


_CITATION = """
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
"""


_DESCRIPTION = """\
Name: NLLB Multi Domain\
NLLB Multi Domain is a set of professionally-translated sentences in News, 
Unscripted informal speech, and Health domains. It is designed to enable 
assessment of out-of-domain performance and to study domain adaptation for 
machine translation. Each domain has approximately 3000 sentences.
"""

languages = ["ayr_Latn", "bho_Deva", "dyu_Latn", "fur_Latn", "rus_Cyrl", "wol_Latn"]

class NLLBMDConfig(datalabs.BuilderConfig):
    """BuilderConfig for NLLB Multi-Domain Config"""

    def __init__(self, domain, pair, **kwargs):
        """BuilderConfig for NLLB Multi-Domain Config.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.domain = domain
        self.pair = pair
        super(NLLBMDConfig, self).__init__(**kwargs)

class NLLBMD(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIGS = []
    for lan in languages:
        for domain in ["chat", "news", "health"]:
            new = NLLBMDConfig(
                name=f"{domain}-{lan}",
                domain=domain,
                pair=f"eng_Latn-{lan}",
                version=datalabs.Version("1.0.0"),
                description=f"eng_Latn-{lan} MT for {domain} domain",
            )
            BUILDER_CONFIGS.append(new)


    def _info(self):
        features = datalabs.Features(
            {
                "translation": datalabs.features.Translation(
                    languages=self.config.pair.split("-")
                )
            }
        )

        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # Homepage of the dataset for documentation
            homepage="https://github.com/facebookresearch/flores/blob/main/nllb_md/README.md",
            # datasets.features.FeatureConnectors
            features=features,
            supervised_keys=None,
            languages=self.config.pair.split("-"),
            task_templates=[
                get_task(TaskType.machine_translation)(
                    translation_column="translation",
                    lang_sub_columns=self.config.pair.split("-"),
                )
            ],
        )

    def _split_generators(self, dl_manager):

        source, target = self.config.pair.split("-")
        domain = self.config.domain
        _URL = "https://tinyurl.com/NLLBMD{}".format(domain)
        print(_URL)
        datadir = dl_manager.download_and_extract(_URL)
        datadir = os.path.join(datadir, "NLLB-MD", self.config.domain)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "source_file": os.path.join(datadir, f"train.eng_Latn-{target}.{source}"),
                    "target_file": os.path.join(datadir, f"train.eng_Latn-{target}.{target}"),
                    "split": "train",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "source_file": os.path.join(datadir, f"test.eng_Latn-{target}.{source}"),
                    "target_file": os.path.join(datadir, f"test.eng_Latn-{target}.{target}"),
                    "split": "test",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "source_file": os.path.join(datadir, f"valid.eng_Latn-{target}.{source}"),
                    "target_file": os.path.join(datadir, f"valid.eng_Latn-{target}.{target}"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, source_file, target_file, split):
        """Yields examples."""
        id_ = 0
        source, target = self.config.pair.split("-")
        with open(source_file, encoding="utf8") as source_csv_file:
            with open(target_file, encoding="utf8") as target_csv_file:
                for source_row, target_row in zip(source_csv_file, target_csv_file):
                    source_row = source_row.strip()
                    target_row = target_row.strip()
                    yield id_, {"translation": {source: source_row, 
                                                target: target_row}}
                    id_ += 1
