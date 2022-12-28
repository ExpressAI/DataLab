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
Name: No Language Left Behind Seed Data\
NLLB Seed is a set of professionally-translated sentences in the Wikipedia 
domain. Data for NLLB-Seed was sampled from Wikimedia's List of articles every 
Wikipedia should have, a collection of topics in different fields of knowledge 
and human activity. NLLB-Seed consists of around six thousand sentences in 39 
languages. NLLB-Seed is meant to be used for training rather than model 
evaluation. Due to this difference, NLLB-Seed does not go through the human 
quality assurance process present in FLORES-200.
"""

_URL = "https://tinyurl.com/NLLBSeed"

languages1 = ["ace_Arab", "ace_Latn", "ary_Arab", "arz_Arab", "bam_Latn",
"ban_Latn", "bho_Deva", "bjn_Arab", "bjn_Latn", "bug_Latn", "crh_Latn", 
"dik_Latn", "dzo_Tibt"] 

languages2 = ["fur_Latn", "fuv_Latn", "grn_Latn", "hne_Deva", "kas_Arab", 
"kas_Deva", "knc_Arab", "knc_Latn", "lij_Latn", "lim_Latn", "lmo_Latn", 
"ltg_Latn", "mag_Deva", "mni_Beng", "mri_Latn", "nus_Latn", "prs_Arab", 
"pbt_Arab", "scn_Latn", "shn_Mymr", "srd_Latn", "szl_Latn", "taq_Latn", 
"taq_Tfng", "tzm_Tfng", "vec_Latn"]

pairs = [
    f"{src}-eng_Latn"
    for src in languages1
] + [
    f"eng_Latn-{tar}"
    for tar in languages2
]


class NLLBSeedConfig(datalabs.BuilderConfig):
    """BuilderConfig for NLLB Seed."""

    def __init__(self, pair, **kwargs):
        """BuilderConfig for NLLB Seed.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.pair = pair
        super(NLLBSeedConfig, self).__init__(**kwargs)


class NLLBSeed(datalabs.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = NLLBSeedConfig
    BUILDER_CONFIGS = [
        NLLBSeedConfig(
            name=pair,
            pair=pair
        )
        for pair in pairs
    ]

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
            homepage="https://github.com/facebookresearch/flores/blob/main/nllb_seed/README.md",
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
        """Returns SplitGenerators."""

        source, target = self.config.pair.split("-")
        dir = dl_manager.download_and_extract(_URL)
        dir = os.path.join(dir, "NLLB-Seed", self.config.pair)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_file": os.path.join(dir, f"{source}"),
                    "target_file": os.path.join(dir, f"{target}"),
                    "split": "train",
                },
            )
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
