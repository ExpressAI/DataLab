"""SPACE, a large-scale opinion summarization benchmark"""
import os
import datalabs
from datalabs.tasks import OpinionSummarization
import json
from datalabs.features.features import Sequence
from datalabs.features.features import Value

_CITATION = """\
@article{angelidis-etal-2021-extractive,
    title = "Extractive Opinion Summarization in Quantized Transformer Spaces",
    author = "Angelidis, Stefanos  and
      Amplayo, Reinald Kim  and
      Suhara, Yoshihiko  and
      Wang, Xiaolan  and
      Lapata, Mirella",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.17",
    doi = "10.1162/tacl_a_00366",
    pages = "277--293",
    abstract = "Abstract We present the Quantized Transformer (QT), an unsupervised system for extractive opinion summarization. QT is inspired by Vector- Quantized Variational Autoencoders, which we repurpose for popularity-driven summarization. It uses a clustering interpretation of the quantized space and a novel extraction algorithm to discover popular opinions among hundreds of reviews, a significant step towards opinion summarization of practical scope. In addition, QT enables controllable summarization without further training, by utilizing properties of the quantized space to extract aspect-specific summaries. We also make publicly available Space, a large-scale evaluation benchmark for opinion summarizers, comprising general and aspect-specific summaries for 50 hotels. Experiments demonstrate the promise of our approach, which is validated by human studies where judges showed clear preference for our method over competitive baselines.",
}
"""

_DESCRIPTION = """\
SPACE is a large-scale opinion summarization benchmark for the evaluation of unsupervised summarizers. SPACE is built on TripAdvisor hotel reviews and includes a training set of approximately 1.1 million reviews for over 11 thousand hotels.
It contains a collection of human-written, abstractive opinion summaries for 50 hotels, including high-level general summaries and aspect summaries for six popular aspects: building, cleanliness, food, location, rooms, and service. 
Every summary is based on 100 input reviews, an order of magnitude increase compared to existing corpora. In total, SPACE contains 1,050 gold standard summaries.
see: https://aclanthology.org/2021.tacl-1.17
"""

_HOMEPAGE = "https://github.com/stangelid/qt"
_ABSTRACT = "summaries"
_ARTICLE = "texts"
_KEY = "aspect"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"

class SpaceConfig(datalabs.BuilderConfig):
    """BuilderConfig for Space."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Space.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SpaceConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class SpaceDataset(datalabs.GeneratorBasedBuilder):
    """Space Dataset."""
    BUILDER_CONFIGS = [SpaceConfig(
            name="opinion",
            version=datalabs.Version("1.0.0"),
            description=f"Space Dataset for unsuprvised opinion summarization",
            task_templates=[OpinionSummarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT, aspect_column=_KEY)])]
    DEFAULT_CONFIG_NAME = "opinion"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: Sequence(Value("string")),
                    _ABSTRACT: Sequence(Value("string")),
                    _KEY: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[OpinionSummarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT, aspect_column=_KEY),
            ],
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(_gdrive_url("1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT"))
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": os.path.join(path, "space_train.json"), "f_id": None, "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": os.path.join(path, "space_summ.json"),
                 "f_id": os.path.join(path, "space_summ_splits.txt"), "split": "val"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": os.path.join(path, "space_summ.json"),
                 "f_id": os.path.join(path, "space_summ_splits.txt"), "split": "test"},
            ),
        ]

    def _generate_examples(self, f_path, f_id, split):
        """Generate Space examples."""
        if split == "train":
            with open(f_path, "r") as f:
                data = json.load(f)
            for i, d in enumerate(data):
                reviews = [" ".join(x["sentences"]) for x in d["reviews"]]
                yield i, {_ARTICLE: reviews, _ABSTRACT: None, _KEY: None}
        else:
            ids = set()
            with open(f_id, "r") as f:
                for line in f:
                    line = line.strip()
                    line = line.split("\t")
                    if split == "val" and line[1] == "dev":
                        ids.add(int(line[0]))
                    elif split == "test" and line[1] == "test":
                        ids.add(int(line[0]))
            with open(f_path, "r") as f:
                data = json.load(f)
            cnt = 0
            for d in data:
                if int(d["entity_id"]) in ids:
                    reviews = [" ".join(x["sentences"]) for x in d["reviews"]]
                    for (k, v) in d["summaries"].items():
                        yield cnt, {_ARTICLE: reviews, _ABSTRACT: v, _KEY: k}
                        cnt += 1

                