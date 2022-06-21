"""En2DeSum: A Cross-lingual Summarization Dataset"""
import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{bai-etal-2021-cross,
    title = "Cross-Lingual Abstractive Summarization with Limited Parallel Resources",
    author = "Bai, Yu  and
      Gao, Yang  and
      Huang, Heyan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.538",
    doi = "10.18653/v1/2021.acl-long.538",
    pages = "6910--6924",
    abstract = "Parallel cross-lingual summarization data is scarce, requiring models to better use the limited available cross-lingual resources. Existing methods to do so often adopt sequence-to-sequence networks with multi-task frameworks. Such approaches apply multiple decoders, each of which is utilized for a specific task. However, these independent decoders share no parameters, hence fail to capture the relationships between the discrete phrases of summaries in different languages, breaking the connections in order to transfer the knowledge of the high-resource languages to low-resource languages. To bridge these connections, we propose a novel Multi-Task framework for Cross-Lingual Abstractive Summarization (MCLAS) in a low-resource setting. Employing one unified decoder to generate the sequential concatenation of monolingual and cross-lingual summaries, MCLAS makes the monolingual summarization task a prerequisite of the CLS task. In this way, the shared decoder learns interactions involving alignments and summary patterns across languages, which encourages attaining knowledge transfer. Experiments on two CLS datasets demonstrate that our model significantly outperforms three baseline models in both low-resource and full-dataset scenarios. Moreover, in-depth analysis on the generated summaries and attention heads verifies that interactions are learned well using MCLAS, which benefits the CLS task under limited parallel resources.",
}
"""

_DESCRIPTION = """\
En2DeSum is a cross-lingual summarization dataset between English and German. The final En2DeSum contains 429,393 training samples, 4,305 validation samples, and 4,099 testing samples.
See: https://aclanthology.org/2021.acl-long.538.pdf
"""

_HOMEPAGE = "https://github.com/ybai-nlp/MCLAS"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class En2DeSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for En2DeSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for En2DeSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(En2DeSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class En2DeSumDataset(datalabs.GeneratorBasedBuilder):
    """En2DeSum Dataset."""

    _FILE_ID = "1E4EDszxkHhL4ovgvYiPKOWz9pCegU0yB"
    BUILDER_CONFIGS = [
        En2DeSumConfig(
            name=f"en-de",
            version=datalabs.Version("1.0.0"),
            description=f"En2DeSum Dataset for crosslingual summarization",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        )
    ]
    DEFAULT_CONFIG_NAME = "en-de"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(f_path, f"./en2de/train.jsonl"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(f_path, f"./en2de/valid.jsonl"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(f_path, f"./en2de/test.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate En2DeSum examples."""
        with open(f_path, encoding="utf-8") as f:
            for (id_, x) in enumerate(f):
                x = json.loads(x)
                yield id_, {_ARTICLE: x["text"], _ABSTRACT: x["summary"]}
