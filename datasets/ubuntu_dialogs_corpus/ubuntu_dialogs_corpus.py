import csv
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{DBLP:journals/corr/LowePSP15,
  author    = {Ryan Lowe and
               Nissan Pow and
               Iulian Serban and
               Joelle Pineau},
  title     = {The Ubuntu Dialogue Corpus: {A} Large Dataset for Research in Unstructured
               Multi-Turn Dialogue Systems},
  journal   = {CoRR},
  volume    = {abs/1506.08909},
  year      = {2015},
  url       = {http://arxiv.org/abs/1506.08909},
  archivePrefix = {arXiv},
  eprint    = {1506.08909},
  timestamp = {Mon, 13 Aug 2018 16:48:23 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LowePSP15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
Ubuntu Dialogue Corpus, a dataset containing almost 1 million multi-turn dialogues, with a total of over 7 million utterances and 100 million words. This provides a unique resource for research into building dialogue managers based on neural language models that can make use of large amounts of unlabeled data. The dataset has both the multi-turn property of conversations in the Dialog State Tracking Challenge datasets, and the unstructured nature of interactions from microblog services such as Twitter.
"""
_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/train.csv"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/valid.csv"
_TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/test.csv"


class UbuntuDialogsCorpusConfig(datalabs.BuilderConfig):
    """BuilderConfig for UbuntuDialogsCorpus."""

    def __init__(self, features, **kwargs):
        """BuilderConfig for UbuntuDialogsCorpus.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """

        super(UbuntuDialogsCorpusConfig, self).__init__(
            version=datalabs.Version("1.0.0"), **kwargs
        )
        self.features = features


class UbuntuDialogsCorpus(datalabs.GeneratorBasedBuilder):

    VERSION = datalabs.Version("1.0.0")
    BUILDER_CONFIGS = [
        UbuntuDialogsCorpusConfig(
            name="train",
            features=["Context", "Utterance", "Label"],
            description="training features",
        ),
        UbuntuDialogsCorpusConfig(
            name="validation_test",
            features=["Context", "Ground Truth Utterance"]
            + ["Distractor_" + str(i) for i in range(9)],
            description="dev and test features",
        ),
    ]

    def _info(self):

        features = {
            feature: datalabs.Value("string") for feature in self.config.features
        }
        if self.config.name == "train":
            features["Label"] = datalabs.Value("int32")
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(features),
            supervised_keys=None,
            homepage="https://github.com/rkadlec/ubuntu-ranking-dataset-creator",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        if self.config.name == "train":
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
                ),
            ]
        else:
            return [
                datalabs.SplitGenerator(
                    name=datalabs.Split.VALIDATION,
                    gen_kwargs={"filepath": validation_path},
                ),
                datalabs.SplitGenerator(
                    name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
                ),
            ]

    def _generate_examples(self, filepath):

        with open(filepath, encoding="utf-8") as f:
            data = csv.DictReader(f)
            for id_, row in enumerate(data):
                yield id_, row
