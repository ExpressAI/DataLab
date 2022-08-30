# coding=utf-8
"""Dataset config script for flores （this code is originally from huggingface, them modified by datalab）"""



"""The FLORES-200 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation"""

import os
import sys
import datalabs

from typing import Union, List, Optional


_CITATION = """
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
@inproceedings{,
  title={The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm\'{a}n, Francisco and Fan, Angela},
  year={2021}
}
@inproceedings{,
  title={Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author={Guzm\'{a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1902.01382},
  year={2019}
}
"""

_DESCRIPTION = """\
The creation of FLORES-200 doubles the existing language coverage of FLORES-101. 
Given the nature of the new languages, which have less standardization and require 
more specialized professional translations, the verification process became more complex. 
This required modifications to the translation workflow. FLORES-200 has several languages 
which were not translated from English. Specifically, several languages were translated 
from Spanish, French, Russian and Modern Standard Arabic. Moreover, FLORES-200 also 
includes two script alternatives for four languages. FLORES-200 consists of translations 
from 842 distinct web articles, totaling 3001 sentences. These sentences are divided 
into three splits: dev, devtest, and test (hidden). On average, sentences are approximately 
21 words long.
"""

_HOMEPAGE = "https://github.com/facebookresearch/flores"

_LICENSE = "CC-BY-SA-4.0"

_LANGUAGES = [
"ace_Arab",  "bam_Latn",  "dzo_Tibt",  "hin_Deva",	"khm_Khmr",  "mag_Deva",  "pap_Latn",  "sot_Latn",	"tur_Latn",
"ace_Latn",  "ban_Latn",  "ell_Grek",  "hne_Deva",	"kik_Latn",  "mai_Deva",  "pbt_Arab",  "spa_Latn",	"twi_Latn",
"acm_Arab",  "bel_Cyrl",  "eng_Latn",  "hrv_Latn",	"kin_Latn",  "mal_Mlym",  "pes_Arab",  "srd_Latn",	"tzm_Tfng",
"acq_Arab",  "bem_Latn",  "epo_Latn",  "hun_Latn",	"kir_Cyrl",  "mar_Deva",  "plt_Latn",  "srp_Cyrl",	"uig_Arab",
"aeb_Arab",  "ben_Beng",  "est_Latn",  "hye_Armn",	"kmb_Latn",  "min_Arab",  "pol_Latn",  "ssw_Latn",	"ukr_Cyrl",
"afr_Latn",  "bho_Deva",  "eus_Latn",  "ibo_Latn",	"kmr_Latn",  "min_Latn",  "por_Latn",  "sun_Latn",	"umb_Latn",
"ajp_Arab",  "bjn_Arab",  "ewe_Latn",  "ilo_Latn",	"knc_Arab",  "mkd_Cyrl",  "prs_Arab",  "swe_Latn",	"urd_Arab",
"aka_Latn",  "bjn_Latn",  "fao_Latn",  "ind_Latn",	"knc_Latn",  "mlt_Latn",  "quy_Latn",  "swh_Latn",	"uzn_Latn",
"als_Latn",  "bod_Tibt",  "fij_Latn",  "isl_Latn",	"kon_Latn",  "mni_Beng",  "ron_Latn",  "szl_Latn",	"vec_Latn",
"amh_Ethi",  "bos_Latn",  "fin_Latn",  "ita_Latn",	"kor_Hang",  "mos_Latn",  "run_Latn",  "tam_Taml",	"vie_Latn",
"apc_Arab",  "bug_Latn",  "fon_Latn",  "jav_Latn",	"lao_Laoo",  "mri_Latn",  "rus_Cyrl",  "taq_Latn",	"war_Latn",
"arb_Arab",  "bul_Cyrl",  "fra_Latn",  "jpn_Jpan",	"lij_Latn",  "mya_Mymr",  "sag_Latn",  "taq_Tfng",	"wol_Latn",
"arb_Latn",  "cat_Latn",  "fur_Latn",  "kab_Latn",	"lim_Latn",  "nld_Latn",  "san_Deva",  "tat_Cyrl",	"xho_Latn",
"ars_Arab",  "ceb_Latn",  "fuv_Latn",  "kac_Latn",	"lin_Latn",  "nno_Latn",  "sat_Olck",  "tel_Telu",	"ydd_Hebr",
"ary_Arab",  "ces_Latn",  "gaz_Latn",  "kam_Latn",	"lit_Latn",  "nob_Latn",  "scn_Latn",  "tgk_Cyrl",	"yor_Latn",
"arz_Arab",  "cjk_Latn",  "gla_Latn",  "kan_Knda",	"lmo_Latn",  "npi_Deva",  "shn_Mymr",  "tgl_Latn",	"yue_Hant",
"asm_Beng",  "ckb_Arab",  "gle_Latn",  "kas_Arab",	"ltg_Latn",  "nso_Latn",  "sin_Sinh",  "tha_Thai",	"zho_Hans",
"ast_Latn",  "crh_Latn",  "glg_Latn",  "kas_Deva",	"ltz_Latn",  "nus_Latn",  "slk_Latn",  "tir_Ethi",	"zho_Hant",
"awa_Deva",  "cym_Latn",  "grn_Latn",  "kat_Geor",	"lua_Latn",  "nya_Latn",  "slv_Latn",  "tpi_Latn",	"zsm_Latn",
"ayr_Latn",  "dan_Latn",  "guj_Gujr",  "kaz_Cyrl",	"lug_Latn",  "oci_Latn",  "smo_Latn",  "tsn_Latn",	"zul_Latn",
"azb_Arab",  "deu_Latn",  "hat_Latn",  "kbp_Latn",	"luo_Latn",  "ory_Orya",  "sna_Latn",  "tso_Latn",
"azj_Latn",  "dik_Latn",  "hau_Latn",  "kea_Latn",	"lus_Latn",  "pag_Latn",  "snd_Arab",  "tuk_Latn",
"bak_Cyrl",  "dyu_Latn",  "heb_Hebr",  "khk_Cyrl",	"lvs_Latn",  "pan_Guru",  "som_Latn",  "tum_Latn"
]

_URL = "https://tinyurl.com/flores200dataset"

_SPLITS = ["dev", "devtest"]

_SENTENCES_PATHS = {
    lang: {
        split: os.path.join("flores200_dataset", split, f"{lang}.{split}")
        for split in _SPLITS
    } for lang in _LANGUAGES
}

_METADATA_PATHS = {
    split: os.path.join("flores200_dataset", f"metadata_{split}.tsv")
    for split in _SPLITS
}

from itertools import permutations

def _pairings(iterable, r=2):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


class Flores200Config(datalabs.BuilderConfig):
    """BuilderConfig for the FLORES-200 dataset."""
    def __init__(self, lang: str, lang2: str = None, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datalabs.Version("1.0.0"), **kwargs)
        self.lang = lang
        self.lang2 = lang2


class Flores200(datalabs.GeneratorBasedBuilder):
    """FLORES-200 dataset."""

    BUILDER_CONFIGS = [
        Flores200Config(
            name=lang,
            description=f"FLORES-200: {lang} subset.",
            lang=lang
        )
        for lang in _LANGUAGES
    ] +  [
        Flores200Config(
            name="all",
            description=f"FLORES-200: all language pairs",
            lang=None
        )
    ] +  [
        Flores200Config(
            name=f"{l1}-{l2}",
            description=f"FLORES-200: {l1}-{l2} aligned subset.",
            lang=l1,
            lang2=l2
        ) for (l1,l2) in _pairings(_LANGUAGES)
    ]

    def _info(self):
        features = {
            "id": datalabs.Value("int32"),
            "URL": datalabs.Value("string"),
            "domain": datalabs.Value("string"),
            "topic": datalabs.Value("string"),
            "has_image": datalabs.Value("int32"),
            "has_hyperlink": datalabs.Value("int32")
        }
        if self.config.name != "all" and "-" not in self.config.name:
            features["sentence"] = datalabs.Value("string")
        elif "-" in self.config.name:
            for lang in [self.config.lang, self.config.lang2]:
                features[f"sentence_{lang}"] = datalabs.Value("string")
        else:
            for lang in _LANGUAGES:
                features[f"sentence_{lang}"] = datalabs.Value("string")
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_URL)

        def _get_sentence_paths(split):
            if isinstance(self.config.lang, str) and isinstance(self.config.lang2, str):
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in (self.config.lang, self.config.lang2)]
            elif isinstance(self.config.lang, str):
                sentence_paths = os.path.join(dl_dir, _SENTENCES_PATHS[self.config.lang][split])
            else:
                sentence_paths = [os.path.join(dl_dir, _SENTENCES_PATHS[lang][split]) for lang in _LANGUAGES]
            return sentence_paths
        return [
            datalabs.SplitGenerator(
                name=split,
                gen_kwargs={
                    "sentence_paths": _get_sentence_paths(split),
                    "metadata_path": os.path.join(dl_dir, _METADATA_PATHS[split]),
                }
            ) for split in _SPLITS
        ]

    def _generate_examples(self, sentence_paths: Union[str, List[str]], metadata_path: str, langs: Optional[List[str]] = None):
        """Yields examples as (key, example) tuples."""
        if isinstance(sentence_paths, str):
            with open(sentence_paths, "r") as sentences_file:
                with open(metadata_path, "r") as metadata_file:
                    metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
                    for id_, (sentence, metadata) in enumerate(
                        zip(sentences_file, metadata_lines)
                    ):
                        sentence = sentence.strip()
                        metadata = metadata.split("\t")
                        yield id_, {
                            "id": id_ + 1,
                            "sentence": sentence,
                            "URL": metadata[0],
                            "domain": metadata[1],
                            "topic": metadata[2],
                            "has_image": 1 if metadata == "yes" else 0,
                            "has_hyperlink": 1 if metadata == "yes" else 0
                        }
        else:
            sentences = {}
            if len(sentence_paths) == len(_LANGUAGES):
                langs = _LANGUAGES
            else:
                langs = [self.config.lang, self.config.lang2]
            for path, lang in zip(sentence_paths, langs):
                with open(path, "r") as sent_file:
                    sentences[lang] = [l.strip() for l in sent_file.readlines()]
            with open(metadata_path, "r") as metadata_file:
                metadata_lines = [l.strip() for l in metadata_file.readlines()[1:]]
            for id_, metadata in enumerate(metadata_lines):
                metadata = metadata.split("\t")
                yield id_, {
                    **{
                        "id": id_ + 1,
                        "URL": metadata[0],
                        "domain": metadata[1],
                        "topic": metadata[2],
                        "has_image": 1 if metadata == "yes" else 0,
                        "has_hyperlink": 1 if metadata == "yes" else 0
                    }, **{
                        f"sentence_{lang}": sentences[lang][id_]
                        for lang in langs
                    }
                }
            


