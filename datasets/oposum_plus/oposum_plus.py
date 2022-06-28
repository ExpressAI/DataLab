"""Oposum+, a large-scale opinion summarization benchmark"""
import os
import datalabs
from datalabs import get_task, TaskType
import json
from datalabs.features.features import Sequence
from datalabs.features.features import Value

_CITATION = """\
@inproceedings{amplayo-etal-2021-aspect,
    title = "Aspect-Controllable Opinion Summarization",
    author = "Amplayo, Reinald Kim  and
      Angelidis, Stefanos  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.528",
    doi = "10.18653/v1/2021.emnlp-main.528",
    pages = "6578--6593",
}
"""

_DESCRIPTION = """\
OPOSUM+ is based on OPOSUM. OPOSUM (Angelidis and Lapata, 2018) is a large corpus of product reviews from six different domains: “laptop bags”, “bluetooth headsets”, “boots”, “keyboards”, “televisions”, and “vacuums”. 
It also includes an evaluation set with extractive general summaries. 
We extended this dataset by (a) adding aspect-specific summaries which are human-written and abstractive following the methodology from Angelidis et al. (2021), and (b) increasing the size of the corpus. 
"""

_HOMEPAGE = "https://github.com/rktamplayo/AceSum"
_ABSTRACT = "summaries"
_ARTICLE = "texts"
_KEY = "aspects"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"

class OposumPlusConfig(datalabs.BuilderConfig):
    """BuilderConfig for Oposum+."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Oposum+.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(OposumPlusConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class OposumPlusDataset(datalabs.GeneratorBasedBuilder):
    """Oposum+ Dataset."""
    BUILDER_CONFIGS = [OposumPlusConfig(
            name="aspect",
            version=datalabs.Version("1.0.0"),
            description=f"Oposum+ Dataset for aspect summarization",
            task_templates=[get_task(TaskType.aspect_summarization)(
                source_column=_ARTICLE, reference_column=_ABSTRACT, aspect_column=_KEY)])]
    DEFAULT_CONFIG_NAME = "aspect"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: Sequence(Value("string")),
                    _ABSTRACT: Sequence(Value("string")),
                    _KEY: Sequence(Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[get_task(TaskType.opinion_summarization)(
                source_column=_ARTICLE, reference_column=_ABSTRACT, aspect_column=_KEY),
            ],
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(_gdrive_url("1HByECivioOXacUqqRg5ViPOYBhGvT3S7"))
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path_1": os.path.join(path, "data/oposum/train.sum.jsonl"),
                "f_path_2": None
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path_1": os.path.join(path, "data/oposum/dev.sum.general.jsonl"),
                "f_path_2": os.path.join(path, "data/oposum/dev.sum.aspect.jsonl")
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path_1": os.path.join(path, "data/oposum/test.sum.general.jsonl"),
                "f_path_2": os.path.join(path, "data/oposum/test.sum.aspect.jsonl")
                }
            ),
        ]

    def _generate_examples(self, f_path_1, f_path_2):
        """Generate Oposum+ examples."""
        is_train = f_path_2 is None
        cnt = 0
        if is_train:
            with open(f_path_1, "r") as f:
                for i, x in enumerate(f):
                    data = json.loads(x)
                    yield i, {
                        _ARTICLE: data["reviews"],
                        _ABSTRACT: [data["summary"]],
                        _KEY: data["keywords"],
                    }
        else:
            with open(f_path_1, "r") as f:
                for i, x in enumerate(f):
                    data = json.loads(x)
                    yield cnt, {
                        _ARTICLE: data["reviews"],
                        _ABSTRACT: data["summary"],
                        _KEY: data["keywords"],
                    }
                    cnt += 1
            with open(f_path_2, "r") as f:
                for j, x in enumerate(f):
                    data = json.loads(x)
                    yield cnt, {
                        _ARTICLE: data["reviews"],
                        _ABSTRACT: data["summary"],
                        _KEY: data["keywords"],
                    }
                    cnt += 1

                