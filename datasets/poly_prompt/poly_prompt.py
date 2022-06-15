# coding=utf-8
# Copyright 2022 The HuggingFace datasets Authors, DataLab authors, and the current dataset script contributor.
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

import json
import os
import textwrap

import datalabs

# TODO(PolyPrompt): BibTeX citation
_CITATION = """
"""

# TODO(xtrem):
_DESCRIPTION = """\
PolyPrompt
"""
TASK2LANGS = {
    "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),
    "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),
    "mlqa": "en,es,de,ar,hi,vi,zh".split(","),
    "marc": "en,de,es,fr,ja,zh".split(","),
    "mldoc": "en,de,es,fr,ja,zh,it,ru".split(","),
    "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
    "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),
    "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(
        ","
    ),
    "xlsum": "en,ar,vi,ko,es,zh,ru,fr,tr,hi,id,fa,pt,mr,th,az,bn,np,sr,sw,ta,te,ur,cy,am,my,gu,ha,ig,pa,si,yo".split(
        ","
    ),
}


_TASKS = ["xquad", "tydiqa", "mlqa", "marc", "mldoc", "pawsx", "xnli"]
_NAMES = []
for task in _TASKS:
    langs = TASK2LANGS[task]
    for lang in langs:
        _NAMES.append(f"{task}.{lang}")

_DESCRIPTIONS = {
    "xquad": textwrap.dedent(
        """\
          XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question
          answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from
          the development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into
          ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. Consequently,
          the dataset is entirely parallel across 11 languages."""
    ),
    "tydiqa": textwrap.dedent(
        """Gold passage task (GoldP): Given a passage that is guaranteed to contain the answer, predict the single contiguous span of characters that answers the question. This is more similar to existing reading comprehension datasets (as opposed to the information-seeking task outlined above). This task is constructed with two goals in mind: (1) more directly comparing with prior work and (2) providing a simplified way for researchers to use TyDi QA by providing compatibility with existing code for SQuAD 1.1, XQuAD, and MLQA. Toward these goals, the gold passage task differs from the primary task in several ways: only the gold answer passage is provided rather than the entire Wikipedia article; unanswerable questions have been discarded, similar to MLQA and XQuAD; we evaluate with the SQuAD 1.1 metrics like XQuAD; and Thai and Japanese are removed since the lack of whitespace breaks some tools. """
    ),
    "mlqa": textwrap.dedent(
        """\
          MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
    MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
    German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
    4 different languages on average."""
    ),
    "marc": textwrap.dedent(
        """MARC (Multilingual Amazon Reviews Corpus) is a multilingual text classification dataset with 6 different languages. 
        Here, we use the binarized classification task that is defined by Keung et al. (2020)."""
    ),
    "mldoc": textwrap.dedent(
        """MLDOC is a multilingual document classification dataset with six topic categories."""
    ),
    "xnli": textwrap.dedent(
        """
          The Cross-lingual Natural Language Inference (XNLI) corpus is a crowd-sourced collection of 5,000 test and 2,500 dev 
          pairs for the MultiNLI corpus. The pairs are annotated with textual entailment and translated into 14 languages: 
          French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu. 
          This results in 112.5k annotated pairs. Each premise can be associated with the corresponding hypothesis in the 15 languages, 
          summing up to more than 1.5M combinations. The corpus is made to evaluate how to perform inference 
          in any language (including low-resources ones like Swahili or Urdu) when only English NLI data is available at training time. 
          One solution is cross-lingual sentence encoding, for which XNLI is an evaluation benchmark."""
    ),
    "pawsx": textwrap.dedent(
        """ This dataset contains 23,659 human translated PAWS evaluation pairs and 3,000 machine translated training pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. All translated pairs are sourced from examples in PAWS-Wiki."""
    ),
    "xlsum": textwrap.dedent(
        """XL-Sum is a multilingual summarization dataset covering 45 low- to high-resource languages. In our poly_prompt, 
        we have used 32 of these languages. iso 639-1 codes are: en,ar,vi,ko,es,zhCN,ru,fr,tr,hi,id,fa,pt,mr,th,az,bn,np,srcy,sw,ta,te,ur,cy,am,my,gu,ha,ig,pa,si,yo."""
    ),
    "panx": textwrap.dedent(
        """\
    The WikiANN dataset (Pan et al. 2017) is a dataset with NER annotations for PER, ORG and LOC. It has been
    constructed using the linked entities in Wikipedia pages for 282 different languages including Danish. The dataset
    can be loaded with the DaNLP package:"""
    ),
}
_CITATIONS = {
    "tydiqa": textwrap.dedent(
        """\
            @article{tydiqa,
              title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
              author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
              year    = {2020},
              journal = {Transactions of the Association for Computational Linguistics}
              }"""
    ),
    "xquad": textwrap.dedent(
        """\
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
    "mlqa": textwrap.dedent(
        """\
          @article{lewis2019mlqa,
          title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
          author={Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
          journal={arXiv preprint arXiv:1910.07475},
          year={2019}"""
    ),
    "marc": textwrap.dedent(
        """\
            @article{keung2020multilingual,
              title={The multilingual Amazon reviews corpus},
              author={Keung, Phillip and Lu, Yichao and Szarvas, Gy{\"o}rgy and Smith, Noah A},
              journal={arXiv preprint arXiv:2010.02573},
              year={2020}
            }"""
    ),
    "mldoc": textwrap.dedent(
        """\
        @article{schwenk2018corpus,
          title={A corpus for multilingual document classification in eight languages},
          author={Schwenk, Holger and Li, Xian},
          journal={arXiv preprint arXiv:1805.09821},
          year={2018}
        }"""
    ),
    "pawsx": textwrap.dedent(
        """\
          @InProceedings{pawsx2019emnlp,
          title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
          author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
          booktitle = {Proc. of EMNLP},
          year = {2019}
        }"""
    ),
    "xnli": textwrap.dedent(
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
    "xlsum": textwrap.dedent(
        """\
         @article{hasan2021xl,
          title={XL-sum: Large-scale multilingual abstractive summarization for 44 languages},
          author={Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md Saiful and Samin, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and Rahman, M Sohel and Shahriyar, Rifat},
          journal={arXiv preprint arXiv:2106.13822},
          year={2021}
    }"""
    ),
    "panx": textwrap.dedent(
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
    "xquad": {
        "id": "samp_id",
        "context": "context",
        "question": "question",
        "answers": "answers",
        "prompt_text": "input",
    },
    "tydiqa": {
        "id": "samp_id",
        "context": "context",
        "question": "question",
        "answers": "answers",
        "prompt_text": "input",
    },
    "mlqa": {
        "id": "samp_id",
        "context": "context",
        "question": "question",
        "answers": "answers",
        "prompt_text": "input",
    },
    "marc": {
        "id": "samp_id",
        "review_title": "review_title",
        "review_body": "review_body",
        "label": "target",
        "prompt_text": "input",
    },
    "mldoc": {
        "id": "samp_id",
        "text": "text",
        "label": "target",
        "prompt_text": "input",
    },
    "pawsx": {
        "id": "samp_id",
        "sentence1": "sent1",
        "sentence2": "sent2",
        "label": "target",
        "prompt_text": "input",
    },
    "xnli": {
        "id": "samp_id",
        "sentence1": "sent1",
        "sentence2": "sent2",
        "label": "target",
        "prompt_text": "input",
    },
}
_DATA_URLS = {
    "tydiqa": "https://s3.amazonaws.com/datalab-hub/poly_prompt/tydiqa/tydiqa.zip",
    "mlqa": "https://s3.amazonaws.com/datalab-hub/poly_prompt/mlqa/mlqa.zip",
    "marc": "https://s3.amazonaws.com/datalab-hub/poly_prompt/marc/marc.zip",
    "mldoc": "https://s3.amazonaws.com/datalab-hub/poly_prompt/mldoc/mldoc.zip",
    "pawsx": "https://s3.amazonaws.com/datalab-hub/poly_prompt/pawsx/pawsx.zip",
    "xquad": "https://s3.amazonaws.com/datalab-hub/poly_prompt/xquad/xquad.zip",
    "xnli": "https://s3.amazonaws.com/datalab-hub/poly_prompt/xnli/xnli.zip",
}


class PolyPromptConfig(datalabs.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, data_url, citation, text_features, **kwargs):
        """
        Args:
            text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
            label_column:
            label_classes
            **kwargs: keyword arguments forwarded to super.
        """
        super(PolyPromptConfig, self).__init__(
            version=datalabs.Version("2.0.0", ""), **kwargs
        )
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation


class PolyPrompt(datalabs.GeneratorBasedBuilder):
    """TODO(PolyPrompt): Short description of my dataset."""

    # TODO(PolyPrompt): Set up version.
    VERSION = datalabs.Version("2.0.0")
    BUILDER_CONFIGS = [
        PolyPromptConfig(
            name=name,
            description=_DESCRIPTIONS[name.split(".")[0]],
            citation=_CITATIONS[name.split(".")[0]],
            text_features=_TEXT_FEATURES[name.split(".")[0]],
            data_url=_DATA_URLS[name.split(".")[0]],
        )
        for name in _NAMES
    ]

    def _info(self):
        features = {
            text_feature: datalabs.Value("string")
            for text_feature in self.config.text_features.keys()
        }
        return datalabs.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description + "\n" + _DESCRIPTION,
            features=datalabs.Features(features),
            languages=[self.config.name.split(".")[-1]],
            supervised_keys=None,
            # Homepage of the dataset for documentation
            citation=self.config.citation + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        task = self.config.name.split(".")[0]
        lang = self.config.name.split(".")[1]
        task_downloaded_files = dl_manager.download_and_extract(self.config.data_url)

        TRAIN = datalabs.SplitGenerator(
            name=datalabs.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": os.path.join(
                    task_downloaded_files, f"{task}/train-{lang}.json"
                )
            },
        )
        TEST = datalabs.SplitGenerator(
            name=datalabs.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": os.path.join(
                    task_downloaded_files, f"{task}/test-{lang}.json"
                )
            },
        )
        RETURNS = [TRAIN, TEST]
        if task not in ["xquad", "mlqa"]:
            VALIDATION = datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        task_downloaded_files, f"{task}/dev-{lang}.json"
                    )
                },
            )
            RETURNS = [TRAIN, VALIDATION, TEST]

        return RETURNS

    def _generate_examples(self, filepath=None, **kwargs):
        """Yields examples."""
        # TODO(PolyPrompt): Yields (key, example) tuples from the dataset
        task = self.config.name.split(".")[0]
        lang = self.config.name.split(".")[1]
        with open(filepath, encoding="utf-8") as f:
            samples = json.load(f)
            feat_dics = _TEXT_FEATURES[task]

            for sample in samples:
                id_ = sample["samp_id"]
                data = {}
                for kfeat, ori_feat in feat_dics.items():
                    data[kfeat] = sample[ori_feat]
                yield id_, data
