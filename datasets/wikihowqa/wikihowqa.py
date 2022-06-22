"""WikiHowQA: Joint Learning of Answer Selection and Answer Summary Generation in Community Question Answering"""
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{Deng2020JointLO,
  title={Joint Learning of Answer Selection and Answer Summary Generation in Community Question Answering},
  author={Yang Deng and Wai Lam and Yuexiang Xie and Daoyuan Chen and Yaliang Li and Min Yang and Ying Shen},
  booktitle={AAAI},
  year={2020}
}
"""

_DESCRIPTION = """\
We present a new CQA corpus, WikiHowQA, for answer summary generation, which contains labels for the answer selection task 
as well as reference summaries for the text summarization task.
see: https://arxiv.org/pdf/1911.09801.pdf
"""

_HOMEPAGE = "https://github.com/dengyang17/wikihowQA"
_LICENSE = "CC-BY-NC-SA"
_ARTICLE = "text"
_ABSTRACT = "summary"


class WikiHowQAConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiHowQA."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WikiHowQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiHowQAConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WikiHowQADataset(datalabs.GeneratorBasedBuilder):
    """WikiHowQA Dataset."""

    BUILDER_CONFIGS = [
        WikiHowQAConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Answer Summarization Dataset.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
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
            license=_LICENSE,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://drive.google.com/u/0/uc?id=1cpd0nXX5d4PbIYOyaDV-BTg3XyruKLXo&export=download&confirm=t"
        f_path = dl_manager.download_and_extract(url)

        train_f_path = os.path.join(f_path, "train.txt")
        valid_f_path = os.path.join(f_path, "valid.txt")
        test_f_path = os.path.join(f_path, "test.txt")
        summary_f_path = os.path.join(f_path, "summary.txt")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path, "summary_f_path": summary_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": valid_f_path, "summary_f_path": summary_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path, "summary_f_path": summary_f_path}
            ),
        ]

    def _generate_examples(self, f_path, summary_f_path):
        """Generate WikiHowQA examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        ids = [line.strip().split("\t")[1] for line in lines]
        ids = list(set(ids))

        summary_f = open(summary_f_path, encoding="utf-8")
        summary_lines = summary_f.readlines()
        datas = []
        for summary_line in summary_lines:
            summary_line = summary_line.strip()
            id = summary_line.split("\t")[0]
            text = summary_line.split("\t")[1]
            summary = summary_line.split("\t")[2]
            if id in ids:
                datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
