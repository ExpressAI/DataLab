"""TODO(xtreme): Add a description here."""


import csv
import json
import os
import textwrap

import datalabs
from datalabs.utils import private_utils

# TODO(xtreme): BibTeX citation
from datalabs import get_task, TaskType

_CITATION = """\
@article{hu2020xtreme,
      author    = {Junjie Hu and Sebastian Ruder and Aditya Siddhant and Graham Neubig and Orhan Firat and Melvin Johnson},
      title     = {XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}
"""

# TODO(xtrem):
_DESCRIPTION = """\
The Cross-lingual TRansfer Evaluation of Multilingual Encoders (XTREME) benchmark is a benchmark for the evaluation of
the cross-lingual generalization ability of pre-trained multilingual models. It covers 40 typologically diverse languages
(spanning 12 language families) and includes nine tasks that collectively require reasoning about different levels of
syntax and semantics. The languages in XTREME are selected to maximize language diversity, coverage in existing tasks,
and availability of training data. Among these are many under-studied languages, such as the Dravidian languages Tamil
(spoken in southern India, Sri Lanka, and Singapore), Telugu and Malayalam (spoken mainly in southern India), and the
Niger-Congo languages Swahili and Yoruba, spoken in Africa.
"""

# Dictionaries of languages should have ISO 639-3 first, then code used by dataset file
_TYDIQA_LANG = {
    "ara": "arabic",
    "ben": "bengali",
    "eng": "english",
    "fin": "finnish",
    "ind": "indonesian",
    "kor": "korean",
    "rus": "russian",
    "swa": "swahili",
    "tel": "telugu",
}
_XNLI_LANG = {
    "ara": "ar",
    "bul": "bg",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "hin": "hi",
    "rus": "ru",
    "swa": "sw",
    "tha": "th",
    "tur": "tr",
    "urd": "ur",
    "vie": "vi",
    "cmn": "zh",
}
_MLQA_LANG = {
    "ara": "ar",
    "deu": "de",
    "vie": "vi",
    "cmn": "zh",
    "eng": "en",
    "spa": "es",
    "hin": "hi",
}
_XQUAD_LANG = {
    "ara": "ar",
    "deu": "de",
    "vie": "vi",
    "cmn": "zh",
    "eng": "en",
    "spa": "es",
    "hin": "hi",
    "ell": "el",
    "rus": "ru",
    "tha": "th",
    "tur": "tr",
}
_PAWSX_LANG = {
    "deu": "de",
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "jpn": "ja",
    "kor": "ko",
    "cmn": "zh",
}
_BUCC_LANG = {"deu": "de", "fra": "fr", "cmn": "zh", "rus": "ru"}
_TATOEBA_LANG = [
    "afr",
    "ara",
    "ben",
    "bul",
    "deu",
    "cmn",
    "ell",
    "est",
    "eus",
    "fin",
    "fra",
    "heb",
    "hin",
    "hun",
    "ind",
    "ita",
    "jav",
    "jpn",
    "kat",
    "kaz",
    "kor",
    "mal",
    "mar",
    "nld",
    "pes",
    "por",
    "rus",
    "spa",
    "swh",
    "tam",
    "tel",
    "tgl",
    "tha",
    "tur",
    "urd",
    "vie",
]

_UD_POS_LANG = {
    "afs": "Afrikaans",
    "ara": "Arabic",
    "eus": "Basque",
    "bul": "Bulgarian",
    "nld": "Dutch",
    "eng": "English",
    "est": "Estonian",
    "fin": "Finnish",
    "fra": "French",
    "deu": "German",
    "ell": "Greek",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hun": "Hungarian",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kaz": "Kazakh",
    "kor": "Korean",
    "cmn": "Chinese",
    "mar": "Marathi",
    "fas": "Persian",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "tgl": "Tagalog",
    "tam": "Tamil",
    "tel": "Telugu",
    "tha": "Thai",
    "tur": "Turkish",
    "urd": "Urdu",
    "vie": "Vietnamese",
    "yor": "Yoruba",
}
_PAN_X_LANG = {
    "afr": "af",
    "ara": "ar",
    "bul": "bg",
    "ben": "bn",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "spa": "es",
    "est": "et",
    "eus": "eu",
    "fas": "fa",
    "fin": "fi",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "hun": "hu",
    "ind": "id",
    "ita": "it",
    "jpn": "ja",
    "jav": "jv",
    "kat": "ka",
    "kaz": "kk",
    "kor": "ko",
    "mal": "ml",
    "mar": "mr",
    "msa": "ms",
    "mya": "my",
    "nld": "nl",
    "por": "pt",
    "rus": "ru",
    "swa": "sw",
    "tam": "ta",
    "tel": "te",
    "tha": "th",
    "tgl": "tl",
    "tur": "tr",
    "urd": "ur",
    "vie": "vi",
    "yor": "yo",
    "cmn": "zh",
}

_NAMES = ["SQuAD.eng"]
for lang3 in _TYDIQA_LANG.keys():
    _NAMES.append(f"tydiqa.{lang3}")
for lang3 in _XNLI_LANG.keys():
    _NAMES.append(f"XNLI.{lang3}")
for lang3 in _PAN_X_LANG.keys():
    _NAMES.append(f"PAN-X.{lang3}")
for lang3 in _MLQA_LANG.keys():
    _NAMES.append(f"MLQA.{lang3}")
for lang3 in _XQUAD_LANG.keys():
    _NAMES.append(f"XQuAD.{lang3}")
for lang3 in _BUCC_LANG.keys():
    _NAMES.append(f"bucc18.{lang3}")
for lang3 in _PAWSX_LANG.keys():
    _NAMES.append(f"PAWS-X.{lang3}")
for lang3 in _TATOEBA_LANG:
    _NAMES.append(f"tatoeba.{lang3}")
for lang3 in _UD_POS_LANG.keys():
    _NAMES.append(f"udpos.{lang3}")

_DESCRIPTIONS = {
    "tydiqa": textwrap.dedent(
        """Gold passage task (GoldP): Given a passage that is guaranteed to contain the
             answer, predict the single contiguous span of characters that answers the question. This is more similar to
             existing reading comprehension datalabs (as opposed to the information-seeking task outlined above).
             This task is constructed with two goals in mind: (1) more directly comparing with prior work and (2) providing
             a simplified way for researchers to use TyDi QA by providing compatibility with existing code for SQuAD 1.1,
             XQuAD, and MLQA. Toward these goals, the gold passage task differs from the primary task in several ways:
             only the gold answer passage is provided rather than the entire Wikipedia article;
             unanswerable questions have been discarded, similar to MLQA and XQuAD;
             we evaluate with the SQuAD 1.1 metrics like XQuAD; and
            Thai and Japanese are removed since the lack of whitespace breaks some tools.
             """
    ),
    "XNLI": textwrap.dedent(
        """
          The Cross-lingual Natural Language Inference (XNLI) corpus is a crowd-sourced collection of 5,000 test and
          2,500 dev pairs for the MultiNLI corpus. The pairs are annotated with textual entailment and translated into
          14 languages: French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese,
          Hindi, Swahili and Urdu. This results in 112.5k annotated pairs. Each premise can be associated with the
          corresponding hypothesis in the 15 languages, summing up to more than 1.5M combinations. The corpus is made to
          evaluate how to perform inference in any language (including low-resources ones like Swahili or Urdu) when only
          English NLI data is available at training time. One solution is cross-lingual sentence encoding, for which XNLI
          is an evaluation benchmark."""
    ),
    "PAWS-X": textwrap.dedent(
        """
          This dataset contains 23,659 human translated PAWS evaluation pairs and 296,406 machine translated training
          pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. All
          translated pairs are sourced from examples in PAWS-Wiki."""
    ),
    "XQuAD": textwrap.dedent(
        """\
          XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question
          answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from
          the development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into
          ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. Consequently,
          the dataset is entirely parallel across 11 languages."""
    ),
    "MLQA": textwrap.dedent(
        """\
          MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
    MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
    German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
    4 different languages on average."""
    ),
    "tatoeba": textwrap.dedent(
        """\
          his data is extracted from the Tatoeba corpus, dated Saturday 2018/11/17.
          For each languages, we have selected 1000 English sentences and their translations, if available. Please check
          this paper for a description of the languages, their families and scripts as well as baseline results.
          Please note that the English sentences are not identical for all language pairs. This means that the results are
          not directly comparable across languages. In particular, the sentences tend to have less variety for several
          low-resource languages, e.g. "Tom needed water", "Tom needs water", "Tom is getting water", ...
                    """
    ),
    "bucc18": textwrap.dedent(
        """Building and Using Comparable Corpora
          """
    ),
    "udpos": textwrap.dedent(
        """\
    Universal Dependencies (UD) is a framework for consistent annotation of grammar (parts of speech, morphological
    features, and syntactic dependencies) across different human languages. UD is an open community effort with over 200
    contributors producing more than 100 treebanks in over 70 languages. If you’re new to UD, you should start by reading
    the first part of the Short Introduction and then browsing the annotation guidelines.
    """
    ),
    "SQuAD": textwrap.dedent(
        """\
    Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
    dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
    articles, where the answer to every question is a segment of text, or span, \
    from the corresponding reading passage, or the question might be unanswerable."""
    ),
    "PAN-X": textwrap.dedent(
        """\
    The WikiANN dataset (Pan et al. 2017) is a dataset with NER annotations for PER, ORG and LOC. It has been
    constructed using the linked entities in Wikipedia pages for 282 different languages including Danish. The dataset
    can be loaded with the DaNLP package:"""
    ),
}
_CITATIONS = {
    "tydiqa": textwrap.dedent(
        (
            """\
            @article{tydiqa,
              title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
              author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
              year    = {2020},
              journal = {Transactions of the Association for Computational Linguistics}
              }"""
        )
    ),
    "XNLI": textwrap.dedent(
        """\
          @InProceedings{conneau2018xnli,
          author = {Conneau, Alexis
                         and Rinott, Ruty
                         and Lample, Guillaume
                         and Williams, Adina
                         and Bowman, Samuel R.
                         and Schwenk, Holger
                         and Stoyanov, Veselin},
          title = {XNLI: Evaluating Cross-lingual Sentence Representations},
          booktitle = {Proceedings of the 2018 Conference on Empirical Methods
                       in Natural Language Processing},
          year = {2018},
          publisher = {Association for Computational Linguistics},
          location = {Brussels, Belgium},
        }"""
    ),
    "XQuAD": textwrap.dedent(
        """
          @article{Artetxe:etal:2019,
              author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
              title     = {On the cross-lingual transferability of monolingual representations},
              journal   = {CoRR},
              volume    = {abs/1910.11856},
              year      = {2019},
              archivePrefix = {arXiv},
              eprint    = {1910.11856}
        }
        """
    ),
    "MLQA": textwrap.dedent(
        """\
          @article{lewis2019mlqa,
          title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
          author={Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
          journal={arXiv preprint arXiv:1910.07475},
          year={2019}"""
    ),
    "PAWS-X": textwrap.dedent(
        """\
          @InProceedings{pawsx2019emnlp,
          title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
          author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
          booktitle = {Proc. of EMNLP},
          year = {2019}
        }"""
    ),
    "tatoeba": textwrap.dedent(
        """\
                    @article{tatoeba,
            title={Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond},
            author={Mikel, Artetxe and Holger, Schwenk,},
            journal={arXiv:1812.10464v2},
            year={2018}
          }"""
    ),
    "bucc18": textwrap.dedent(
        """\
        @inproceedings{zweigenbaum2018overview,
          title={Overview of the third BUCC shared task: Spotting parallel sentences in comparable corpora},
          author={Zweigenbaum, Pierre and Sharoff, Serge and Rapp, Reinhard},
          booktitle={Proceedings of 11th Workshop on Building and Using Comparable Corpora},
          pages={39--42},
          year={2018}
        }"""
    ),
    "udpos": textwrap.dedent(
        """\
        @inproceedings{nivre2016universal,
          title={Universal dependencies v1: A multilingual treebank collection},
          author={Nivre, Joakim and De Marneffe, Marie-Catherine and Ginter, Filip and Goldberg, Yoav and Hajic, Jan and Manning, Christopher D and McDonald, Ryan and Petrov, Slav and Pyysalo, Sampo and Silveira, Natalia and others},
          booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
          pages={1659--1666},
          year={2016}
        }"""
    ),
    "SQuAD": textwrap.dedent(
        """\
        @article{2016arXiv160605250R,
           author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                     Konstantin and {Liang}, Percy},
            title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
          journal = {arXiv e-prints},
             year = 2016,
              eid = {arXiv:1606.05250},
            pages = {arXiv:1606.05250},
            archivePrefix = {arXiv},
           eprint = {1606.05250},
}"""
    ),
    "PAN-X": textwrap.dedent(
        """\
                    @article{pan-x,
            title={Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond},
            author={Xiaoman, Pan and Boliang, Zhang and Jonathan, May and Joel, Nothman and Kevin, Knight and Heng, Ji},
            volume={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers}
            year={2017}
          }"""
    ),
}

_TEXT_FEATURES = {
    "XNLI": {
        "language": "language",
        "sentence1": "sentence1",
        "sentence2": "sentence2",
    },
    "tydiqa": {
        "id": "id",
        "title": "title",
        "context": "context",
        "question": "question",
        "answers": "answers",
    },
    "XQuAD": {
        "id": "id",
        "context": "context",
        "question": "question",
        "answers": "answers",
    },
    "MLQA": {
        "id": "id",
        "title": "title",
        "context": "context",
        "question": "question",
        "answers": "answers",
    },
    "tatoeba": {
        "source_sentence": "",
        "target_sentence": "",
        "source_lang": "",
        "target_lang": "",
    },
    "bucc18": {
        "source_sentence": "",
        "target_sentence": "",
        "source_lang": "",
        "target_lang": "",
    },
    "PAWS-X": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"},
    "udpos": {"tokens": "", "tags": ""},
    "SQuAD": {
        "id": "id",
        "title": "title",
        "context": "context",
        "question": "question",
        "answers": "answers",
    },
    "PAN-X": {"tokens": "", "tags": "", "lang": ""},
}
_DATA_URLS = {
    "tydiqa": "https://storage.googleapis.com/tydiqa/",
    "XNLI": "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip",
    "XQuAD": "https://github.com/deepmind/xquad/raw/master/",
    "MLQA": "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip",
    "PAWS-X": "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz",
    "bucc18": "https://comparable.limsi.fr/bucc2018/",
    "tatoeba": "https://github.com/facebookresearch/LASER/raw/main/data/tatoeba/v1/",
    "udpos": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz",
    "SQuAD": "https://rajpurkar.github.io/SQuAD-explorer/dataset/",
    "PAN-X": "https://s3.amazonaws.com/datasets.huggingface.co/wikiann/1.1.0/panx_dataset.zip",
}

_URLS = {
    "tydiqa": "https://github.com/google-research-datasets/tydiqa",
    "XQuAD": "https://github.com/deepmind/xquad",
    "XNLI": "https://www.nyu.edu/projects/bowman/xnli/",
    "MLQA": "https://github.com/facebookresearch/MLQA",
    "PAWS-X": "https://github.com/google-research-datasets/paws/tree/master/pawsx",
    "bucc18": "https://comparable.limsi.fr/bucc2018/",
    "tatoeba": "https://github.com/facebookresearch/LASER/blob/main/data/tatoeba/v1/README.md",
    "udpos": "https://universaldependencies.org/",
    "SQuAD": "https://rajpurkar.github.io/SQuAD-explorer/",
    "PAN-X": "https://github.com/afshinrahimi/mmner",
}


class XtremeConfig(datalabs.BuilderConfig):
    """BuilderConfig for Xtreme"""

    def __init__(self, data_url, citation, url, text_features, **kwargs):
        """
        Args:
            text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
            label_column:
            label_classes
            **kwargs: keyword arguments forwarded to super.
        """
        super(XtremeConfig, self).__init__(
            version=datalabs.Version("1.0.0", ""), **kwargs
        )
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation
        self.url = url


class Xtreme(datalabs.GeneratorBasedBuilder):
    """TODO(xtreme): Short description of my dataset."""

    # TODO(xtreme): Set up version.
    VERSION = datalabs.Version("0.1.0")
    BUILDER_CONFIGS = [
        XtremeConfig(
            name=name,
            description=_DESCRIPTIONS[name.split(".")[0]],
            citation=_CITATIONS[name.split(".")[0]],
            text_features=_TEXT_FEATURES[name.split(".")[0]],
            data_url=_DATA_URLS[name.split(".")[0]],
            url=_URLS[name.split(".")[0]],
        )
        for name in _NAMES
    ]

    def _info(self):
        features = {
            text_feature: datalabs.Value("string")
            for text_feature in self.config.text_features.keys()
        }
        languages = self.config.name.split(".")[1:]
        if "answers" in features.keys():
            features["answers"] = datalabs.features.Sequence(
                {
                    "answer_start": datalabs.Value("int32"),
                    "text": datalabs.Value("string"),
                }
            )
            task_template = get_task(TaskType.qa_extractive)(
                question_column="question",
                context_column="context",
                answers_column="answers",
            )
        if self.config.name.startswith("PAWS-X"):
            features = PawsxParser.features
            task_template = get_task(TaskType.paraphrase_identification)(
                text1_column="sentence1", text2_column="sentence2", label_column="label"
            )
        elif self.config.name.startswith("XNLI"):
            features["gold_label"] = datalabs.features.ClassLabel(
                names=[
                    "entailment",
                    "contradiction",
                    "neutral",
                ]
            )
            task_template = get_task(TaskType.natural_language_inference)(
                text1_column="sentence1",
                text2_column="sentence2",
                label_column="gold_label",
            )
        elif self.config.name.startswith("udpos"):
            features = UdposParser.features
            task_template = get_task(TaskType.part_of_speech)(
                tokens_column="tokens",
                tags_column="tags",
            )
        elif self.config.name.startswith("PAN-X"):
            features = PanxParser.features
            task_template = get_task(TaskType.named_entity_recognition)(
                tokens_column="tokens",
                tags_column="tags",
            )
        elif self.config.name.startswith("bucc18") or self.config.name.startswith(
            "tatoeba"
        ):
            features["answers"] = [features.pop("target_sentence")]
            task_template = get_task(TaskType.retrieval)(
                query_column="source_sentence",
                answers_column="answers",
            )
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datalabs page.
            description=self.config.description + "\n" + _DESCRIPTION,
            # datalabs.features.FeatureConnectors
            features=datalabs.Features(features),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/google-research/xtreme"
            + "\t"
            + self.config.url,
            citation=self.config.citation + "\n" + _CITATION,
            languages=languages,
            task_templates=[task_template],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name.startswith("tydiqa"):
            train_url = "v1.1/tydiqa-goldp-v1.1-train.json"
            dev_url = "v1.1/tydiqa-goldp-v1.1-dev.json"
            urls_to_download = {
                "train": self.config.data_url + train_url,
                "dev": self.config.data_url + dev_url,
            }
            dl_dir = dl_manager.download_and_extract(urls_to_download)
            split_gens = [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": dl_dir["train"]},
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": dl_dir["dev"]},
                ),
            ]
            if private_utils.has_private_loc():
                test_url = f"{private_utils.PRIVATE_LOC}/xtreme/tydiqa-goldp-v1.1-test.json"
                test_path = dl_manager.download_and_extract(test_url)
                split_gens.append(
                    datalabs.SplitGenerator(
                        name=datalabs.Split.TEST,
                        # These kwargs will be passed to _generate_examples
                        gen_kwargs={"filepath": test_path},
                    ),
                )
            return split_gens
        if self.config.name.startswith("XNLI"):
            dl_dir = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(dl_dir, "XNLI-1.0")
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    gen_kwargs={"filepath": os.path.join(data_dir, "xnli.test.tsv")},
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={"filepath": os.path.join(data_dir, "xnli.dev.tsv")},
                ),
            ]

        if self.config.name.startswith("MLQA"):
            mlqa_downloaded_files = dl_manager.download_and_extract(
                self.config.data_url
            )
            lang3 = self.config.name.split(".")[-1]
            lang2 = _MLQA_LANG[lang3]
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(mlqa_downloaded_files, "MLQA_V1/test"),
                            f"test-context-{lang2}-question-{lang2}.json",
                        )
                    },
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(mlqa_downloaded_files, "MLQA_V1/dev"),
                            f"dev-context-{lang2}-question-{lang2}.json",
                        )
                    },
                ),
            ]

        if self.config.name.startswith("XQuAD"):
            lang3 = self.config.name.split(".")[1]
            lang2 = _XQUAD_LANG[lang3]
            xquad_downloaded_file = dl_manager.download_and_extract(
                self.config.data_url + f"xquad.{lang2}.json"
            )
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": xquad_downloaded_file},
                ),
            ]
        if self.config.name.startswith("PAWS-X"):
            return PawsxParser.split_generators(
                dl_manager=dl_manager, config=self.config
            )
        elif self.config.name.startswith("tatoeba"):
            lang3 = self.config.name.split(".")[1]

            tatoeba_source_data = dl_manager.download_and_extract(
                self.config.data_url + f"tatoeba.{lang3}-eng.{lang3}"
            )
            tatoeba_eng_data = dl_manager.download_and_extract(
                self.config.data_url + f"tatoeba.{lang3}-eng.eng"
            )
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": (tatoeba_source_data, tatoeba_eng_data)},
                ),
            ]
        if self.config.name.startswith("bucc18"):
            lang3 = self.config.name.split(".")[1]
            lang2 = _BUCC_LANG[lang3]
            bucc18_dl_test_archive = dl_manager.download(
                self.config.data_url + f"bucc2018-{lang2}-en.training-gold.tar.bz2"
            )
            bucc18_dl_dev_archive = dl_manager.download(
                self.config.data_url + f"bucc2018-{lang2}-en.sample-gold.tar.bz2"
            )
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": dl_manager.iter_archive(bucc18_dl_dev_archive)
                    },
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST,
                    gen_kwargs={
                        "filepath": dl_manager.iter_archive(bucc18_dl_test_archive)
                    },
                ),
            ]
        if self.config.name.startswith("udpos"):
            return UdposParser.split_generators(
                dl_manager=dl_manager, config=self.config
            )

        if self.config.name.startswith("SQuAD"):

            urls_to_download = {
                "train": self.config.data_url + "train-v1.1.json",
                "dev": self.config.data_url + "dev-v1.1.json",
            }
            downloaded_files = dl_manager.download_and_extract(urls_to_download)

            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded_files["train"]},
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded_files["dev"]},
                ),
            ]

        if self.config.name.startswith("PAN-X"):
            return PanxParser.split_generators(
                dl_manager=dl_manager, config=self.config
            )

    def _generate_examples(self, filepath=None, **kwargs):
        """Yields examples."""
        # TODO(xtreme): Yields (key, example) tuples from the dataset

        if self.config.name.startswith("MLQA") or self.config.name.startswith("SQuAD"):
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for article in data["data"]:
                    title = article.get("title", "").strip()
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"].strip()
                        for qa in paragraph["qas"]:
                            question = qa["question"].strip()
                            id_ = qa["id"]

                            answer_starts = [
                                answer["answer_start"] for answer in qa["answers"]
                            ]
                            answers = [
                                answer["text"].strip() for answer in qa["answers"]
                            ]

                            # Features currently used are "context", "question", and "answers".
                            # Others are extracted here for the ease of future expansions.
                            yield id_, {
                                "title": title,
                                "context": context,
                                "question": question,
                                "id": id_,
                                "answers": {
                                    "answer_start": answer_starts,
                                    "text": answers,
                                },
                            }
        elif self.config.name.startswith("tydiqa"):
            lang3 = self.config.name.split(".")[-1]
            langfull = _TYDIQA_LANG[lang3]
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for article in data["data"]:
                    title = article.get("title", "").strip()
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"].strip()
                        for qa in paragraph["qas"]:
                            id_ = qa["id"]
                            examp_langfull = id_.split("-")[0]
                            if examp_langfull == langfull:
                                question = qa["question"].strip()

                                answer_starts = [
                                    answer["answer_start"] for answer in qa["answers"]
                                ]
                                answers = [
                                    answer["text"].strip() for answer in qa["answers"]
                                ]

                                # Features currently used are "context", "question", and "answers".
                                # Others are extracted here for the ease of future expansions.
                                yield id_, {
                                    "title": title,
                                    "context": context,
                                    "question": question,
                                    "id": id_,
                                    "answers": {
                                        "answer_start": answer_starts,
                                        "text": answers,
                                    },
                                }
        elif self.config.name.startswith("XNLI"):
            lang3 = self.config.name.split(".")[-1]
            lang2 = _XNLI_LANG[lang3]
            with open(filepath, encoding="utf-8") as f:
                data = csv.DictReader(f, delimiter="\t")
                for id_, row in enumerate(data):
                    if row["language"] == lang2:
                        yield id_, {
                            "sentence1": row["sentence1"],
                            "sentence2": row["sentence2"],
                            "language": lang3,
                            "gold_label": row["gold_label"],
                        }
        elif self.config.name.startswith("PAWS-X"):
            yield from PawsxParser.generate_examples(
                config=self.config, filepath=filepath, **kwargs
            )
        elif self.config.name.startswith("XQuAD"):
            with open(filepath, encoding="utf-8") as f:
                xquad = json.load(f)
                for article in xquad["data"]:
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"].strip()
                        for qa in paragraph["qas"]:
                            question = qa["question"].strip()
                            id_ = qa["id"]

                            answer_starts = [
                                answer["answer_start"] for answer in qa["answers"]
                            ]
                            answers = [
                                answer["text"].strip() for answer in qa["answers"]
                            ]

                            # Features currently used are "context", "question", and "answers".
                            # Others are extracted here for the ease of future expansions.
                            yield id_, {
                                "context": context,
                                "question": question,
                                "id": id_,
                                "answers": {
                                    "answer_start": answer_starts,
                                    "text": answers,
                                },
                            }
        elif self.config.name.startswith("bucc18"):
            lang3 = self.config.name.split(".")[1]
            lang2 = _BUCC_LANG[lang3]
            lang2to3 = {v: k for k, v in _BUCC_LANG.items()}
            lang2to3["en"] = "eng"
            data_dir = f"bucc2018/{lang2}-en"
            for path, file in filepath:
                if path.startswith(data_dir):
                    csv_content = [line.decode("utf-8") for line in file]
                    if path.endswith("en"):
                        target_id_map = {
                            k: v for k, v in csv.reader(csv_content, delimiter="\t")
                        }
                    elif path.endswith("gold"):
                        source_target_ids = list(
                            csv.reader(csv_content, delimiter="\t")
                        )
                    else:
                        source_id_map = {
                            k: v for k, v in csv.reader(csv_content, delimiter="\t")
                        }
            for id_, pair in enumerate(source_target_ids):
                source_sent = source_id_map.get(pair[0])
                target_sent = target_id_map.get(pair[1])
                yield id_, {
                    "source_sentence": source_sent,
                    "answers": [target_sent],
                    "source_lang": lang2to3[pair[0].split("-")[0]],
                    "target_lang": lang2to3[pair[1].split("-")[0]],
                }
        elif self.config.name.startswith("tatoeba"):
            source_file = filepath[0]
            target_file = filepath[1]
            source_sentences = []
            target_sentences = []
            with open(source_file, encoding="utf-8") as f1:
                for row in f1:
                    source_sentences.append(row)
            with open(target_file, encoding="utf-8") as f2:
                for row in f2:
                    target_sentences.append(row)
            for i in range(len(source_sentences)):
                yield i, {
                    "source_sentence": source_sentences[i],
                    "answers": [target_sentences[i]],
                    "source_lang": source_file.split(".")[-1],
                    "target_lang": "eng",
                }
        elif self.config.name.startswith("udpos"):
            yield from UdposParser.generate_examples(
                config=self.config, filepath=filepath, **kwargs
            )
        elif self.config.name.startswith("PAN-X"):
            yield from PanxParser.generate_examples(filepath=filepath, **kwargs)
        else:
            raise ValueError(f"Bad config name {self.config.name}")


class PanxParser:

    features = datalabs.Features(
        {
            "tokens": datalabs.Sequence(datalabs.Value("string")),
            "tags": datalabs.Sequence(
                datalabs.features.ClassLabel(
                    names=[
                        "O",
                        "B-PER",
                        "I-PER",
                        "B-ORG",
                        "I-ORG",
                        "B-LOC",
                        "I-LOC",
                    ]
                )
            ),
            "langs": datalabs.Sequence(datalabs.Value("string")),
        }
    )

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        data_dir = dl_manager.download_and_extract(config.data_url)
        lang3 = config.name.split(".")[1]
        lang2 = _PAN_X_LANG[lang3]
        archive = os.path.join(data_dir, lang2 + ".tar.gz")
        split_filenames = {
            datalabs.Split.TRAIN: "train",
            datalabs.Split.VALIDATION: "dev",
            datalabs.Split.TEST: "test",
        }
        return [
            datalabs.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": dl_manager.iter_archive(archive),
                    "filename": split_filenames[split],
                },
            )
            for split in split_filenames
        ]

    @staticmethod
    def generate_examples(filepath=None, filename=None):
        lang2to3 = {v: k for k, v in _PAN_X_LANG.items()}
        idx = 1
        for path, file in filepath:
            if path.endswith(filename):
                tokens = []
                ner_tags = []
                langs = []
                for line in file:
                    line = line.decode("utf-8")
                    if line == "" or line == "\n":
                        if tokens:
                            yield idx, {
                                "tokens": tokens,
                                "tags": ner_tags,
                                "langs": langs,
                            }
                            idx += 1
                            tokens = []
                            ner_tags = []
                            langs = []
                    else:
                        # pan-x data is tab separated
                        splits = line.split("\t")
                        # strip out en: prefix
                        langs.append(lang2to3[splits[0][:2]])
                        tokens.append(splits[0][3:])
                        if len(splits) > 1:
                            ner_tags.append(splits[-1].replace("\n", ""))
                        else:
                            # examples have no label in test set
                            ner_tags.append("O")
                if tokens:
                    yield idx, {
                        "tokens": tokens,
                        "tags": ner_tags,
                        "langs": langs,
                    }


class PawsxParser:

    features = datalabs.Features(
        {
            "id": datalabs.Value("int32"),
            "sentence1": datalabs.Value("string"),
            "sentence2": datalabs.Value("string"),
            "label": datalabs.features.ClassLabel(
                names=[
                    "0",
                    "1",
                ]
            ),
        }
    )

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        lang3 = config.name.split(".")[1]
        archive = dl_manager.download(config.data_url)
        split_filenames = {
            datalabs.Split.TRAIN: "translated_train.tsv"
            if lang3 != "eng"
            else "train.tsv",
            datalabs.Split.VALIDATION: "dev_2k.tsv",
            datalabs.Split.TEST: "test_2k.tsv",
        }
        return [
            datalabs.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": dl_manager.iter_archive(archive),
                    "filename": split_filenames[split],
                },
            )
            for split in split_filenames
        ]

    @staticmethod
    def generate_examples(config=None, filepath=None, filename=None):
        lang3 = config.name.split(".")[1]
        lang2 = _PAWSX_LANG[lang3]
        for path, file in filepath:
            if f"/{lang2}/" in path and path.endswith(filename):
                lines = (line.decode("utf-8").replace("\\t", "\t") for line in file)
                # skip header
                next(lines)
                for id_, line in enumerate(lines):
                    row = line.split("\t")
                    if len(row) == 4:
                        yield id_, {
                            "id": int(row[0]),
                            "sentence1": row[1],
                            "sentence2": row[2],
                            "label": row[3],
                        }
                    else:
                        raise ValueError(f"Invalid row: {row}")


class UdposParser:

    features = datalabs.Features(
        {
            "tokens": datalabs.Sequence(datalabs.Value("string")),
            "tags": datalabs.Sequence(
                datalabs.features.ClassLabel(
                    names=[
                        "ADJ",
                        "ADP",
                        "ADV",
                        "AUX",
                        "CCONJ",
                        "DET",
                        "INTJ",
                        "NOUN",
                        "NUM",
                        "PART",
                        "PRON",
                        "PROPN",
                        "PUNCT",
                        "SCONJ",
                        "SYM",
                        "VERB",
                        "X",
                    ]
                )
            ),
        }
    )

    @staticmethod
    def split_generators(dl_manager=None, config=None):
        archive = dl_manager.download(config.data_url)
        split_names = {
            datalabs.Split.TRAIN: "train",
            datalabs.Split.VALIDATION: "dev",
            datalabs.Split.TEST: "test",
        }
        split_generators = {
            split: datalabs.SplitGenerator(
                name=split,
                gen_kwargs={
                    "filepath": dl_manager.iter_archive(archive),
                    "split": split_names[split],
                },
            )
            for split in split_names
        }
        lang3 = config.name.split(".")[1]
        langfull = _UD_POS_LANG[lang3]
        if lang3 in ["tgl", "tha", "yor"]:
            return [split_generators["test"]]
        elif lang3 == "kaz":
            return [split_generators["train"], split_generators["test"]]
        else:
            return [
                split_generators["train"],
                split_generators["validation"],
                split_generators["test"],
            ]

    @staticmethod
    def generate_examples(config=None, filepath=None, split=None):
        lang3 = config.name.split(".")[1]
        langfull = _UD_POS_LANG[lang3]
        idx = 0
        for path, file in filepath:
            if f"_{langfull}" in path and split in path and path.endswith(".conllu"):
                # For lang other than [see below], we exclude Arabic-NYUAD which does not contains any words, only _
                if lang3 in ["kaz", "tgl", "tha", "yor"] or "NYUAD" not in path:
                    lines = (line.decode("utf-8") for line in file)
                    data = csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
                    tokens = []
                    pos_tags = []
                    for id_row, row in enumerate(data):
                        if len(row) >= 10 and row[1] != "_" and row[3] != "_":
                            tokens.append(row[1])
                            pos_tags.append(row[3])
                        if len(row) == 0 and len(tokens) > 0:
                            yield idx, {
                                "tokens": tokens,
                                "tags": pos_tags,
                            }
                            idx += 1
                            tokens = []
                            pos_tags = []
