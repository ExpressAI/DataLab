import json
import os

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{Shah2021NutribulletsSH,
  title={Nutri-bullets: Summarizing Health Studies by Composing Segments},
  author={Darsh J. Shah and L. Yu and Tao Lei and Regina Barzilay},
  booktitle={AAAI},
  year={2021}
}
"""

_DESCRIPTION = """\
Healthline dataset consists of scientific abstracts as inputs and human written summaries as outputs.
Domain experts curate summaries for a general audience in the Healthline dataset.
These summaries describe nutrition and health benefits of a specific food. 
In the HealthLine dataset, each food has multiple bullet summaries, where each bullet typically talks about a different health impact (hydration, anti-diabetic etc).
see: https://ojs.aaai.org/index.php/AAAI/article/view/17624/17431
"""

_HOMEPAGE = "https://github.com/darsh10/Nutribullets"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class HealthlineConfig(datalabs.BuilderConfig):
    """BuilderConfig for Healthline."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Healthline.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HealthlineConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class HealthlineDataset(datalabs.GeneratorBasedBuilder):
    """Healthline Dataset."""

    _FILE_ID = "18ZU5VuiPC329c_BbyBLh2_m67vQ-jE3Y"
    BUILDER_CONFIGS = [
        HealthlineConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description=f"Healthline Dataset for summarization",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

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
        path = dl_manager.download(_gdrive_url(self._FILE_ID))
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": path,
                    "split": "train",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": path,
                    "split": "dev",
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, f_path, split):
        """Generate Healthline examples."""
        with open(f_path, encoding="utf-8") as f:
            data = json.load(f)
        id_ = 0
        for k in data:
            if "split" not in data[k]:
                continue
            if data[k]["split"] == split:
                x = data[k]
                if "summary_inputs" not in x:
                    continue
                if "summary_pubmed_articles" not in x["summary_inputs"]:
                    continue
                summaries, article_ids = [], []
                for (summary, id) in x["summary_inputs"][
                    "summary_pubmed_articles"
                ].items():
                    summaries.append(summary.strip())
                    article_ids.append(id)
                if "pubmed_sentences" not in x:
                    continue
                for (s, id) in zip(summaries, article_ids):
                    text = []
                    for i in id:
                        if i in x["pubmed_sentences"]:
                            for a in x["pubmed_sentences"][i]:
                                text.append(" ".join(a[0]))
                    yield id_, {_ARTICLE: " ".join(text), _ABSTRACT: s}
                    id_ += 1
