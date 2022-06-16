"""Welsh Summarization Dataset."""
import os
import pickle

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{Ezeani2022IntroducingTW,
  title={Introducing the Welsh Text Summarisation Dataset and Baseline Systems},
  author={Ignatius M Ezeani and Mahmoud El-Haj and Jonathan Morris and Dawn Knight},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.02545}
}
"""

_DESCRIPTION = """\
This paper introduces the first Welsh summarization dataset.
The dataset was created by Welsh speakers through manually summarising Welsh Wikipedia articles.
see: https://arxiv.org/pdf/2205.02545.pdf
"""

_HOMEPAGE = "https://github.com/UCREL/welsh-summarization-dataset"
_LICENSE = "Creative Commons Zero v1.0 Universal"
_ARTICLE = "text"
_ABSTRACT = "summary"


class WelshConfig(datalabs.BuilderConfig):
    """BuilderConfig for Welsh."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Welsh.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WelshConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WelshDataset(datalabs.GeneratorBasedBuilder):
    """Welsh Dataset."""

    BUILDER_CONFIGS = [
        WelshConfig(
            name="wiki-ref",
            version=datalabs.Version("1.0.0"),
            description="Welsh summarization dataset. Single wiki reference.",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        WelshConfig(
            name="human-ref",
            version=datalabs.Version("1.0.0"),
            description="Welsh summarization dataset. Multiple human written references.",
            task_templates=[
                get_task(TaskType.multi_ref_summarization)(
                    source_column=_ARTICLE, reference_column="summaries"
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "human-ref"

    def _info(self):
        if self.config.name == "wiki-ref":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(
                    features_sample, get_schema_of_sample_level_features
                )
        elif self.config.name == "human-ref":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    "summaries": datalabs.Sequence(datalabs.Value("string")),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            license=_LICENSE,
            languages=["cy"],  # https://huggingface.co/languages#:~:text=38-,Welsh,-cy
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        url = "https://github.com/UCREL/welsh-summarization-dataset/archive/refs/heads/main.zip"
        f_path = dl_manager.download_and_extract(url)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path, "welsh-summarization-dataset-main/data/dataset.pkl"
                    )
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate Welsh examples."""
        with open(f_path, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        dataset = dataset.to_dict()

        fileid = dataset["fileId"]
        article = dataset["article"]
        wiki_summary = dataset["wiki_summary"]
        human_summary = dataset["human_summary"]

        final_dataset = {}
        for (id, text, w_summary, h_summary) in zip(
            fileid.values(),
            article.values(),
            wiki_summary.values(),
            human_summary.values(),
        ):
            if id not in final_dataset:
                final_dataset[id] = [
                    {
                        "article": text,
                        "wiki_summary": w_summary,
                        "human_summary": h_summary,
                    }
                ]
            else:
                final_dataset[id].append(
                    {
                        "article": text,
                        "wiki_summary": w_summary,
                        "human_summary": h_summary,
                    }
                )

        datas = []
        for samples in final_dataset.values():
            text = samples[0]["article"].replace("\n", "")
            wiki_summary = samples[0]["wiki_summary"].replace("\n", "")
            human_summaries = [
                sample["human_summary"].replace("\n", "") for sample in samples
            ]
            if self.config.name == "human-ref":
                datas.append((text, human_summaries))
            else:
                datas.append((text, wiki_summary))

        for id_, (text, summary) in enumerate(datas):
            if self.config.name == "human-ref":
                raw_feature_info = {_ARTICLE: text, "summaries": summary}
            else:
                raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
