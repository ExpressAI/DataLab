import json
import os
import datalabs
from datalabs import get_task, TaskType
from nltk import word_tokenize


_DESCRIPTION = """
 We hire highly-qualified contractors to read stories and write original summaries from scratch. To amortize reading time,
 we collect five summaries per document, with the first giving an overview and the subsequent
 four addressing specific questions. We use this protocol to collect SQuALITY, a dataset of question-focused summaries built on the same
 public-domain short stories as the multiplechoice dataset QuALITY
"""

_CITATION = """\
    @article{Wang2022SQuALITYBA,
      title={SQuALITY: Building a Long-Document Summarization Dataset the Hard Way},
      author={Alex Wang and Richard Yuanzhe Pang and Angelica Chen and Jason Phang and Samuel R. Bowman},
      journal={ArXiv},
      year={2022},
      volume={abs/2205.11465}
    }
"""

_ABSTRACT = "summaries"
_ARTICLE = "text"
_KEY = "query"
_HOMEPAGE = "https://github.com/nyu-mll/SQuALITY"
_LICENSE = "The Project Gutenberg License (https://www.gutenberg.org/policy/license.html) for stories and CC BY license for summaries."

class SQuALITYConfig(datalabs.BuilderConfig):
    """BuilderConfig for SQuALITY."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for SQuALITY.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SQuALITYConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class SQuALITYDataset(datalabs.GeneratorBasedBuilder):
    """SQuALITY Dataset."""
    _URL = {
        "v1": {
            "train": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1/train.jsonl",
            "val": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1/dev.jsonl",
            "test": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1/test.jsonl",
        },
        "v1.1": {
            "train": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1-1/train.jsonl",
            "val": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1-1/dev.jsonl",
            "test": "https://raw.githubusercontent.com/nyu-mll/SQuALITY/main/data/v1-1/test.jsonl",
        }
    }
    BUILDER_CONFIGS = [
        SQuALITYConfig(
            name="v1",
            version=datalabs.Version("1.0.0"),
            description="SQuALITY dataset for summarization, v1 summarization version",
            task_templates=[get_task(TaskType.multi_ref_query_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column=_KEY)]
        ),
        SQuALITYConfig(
            name="v1.1",
            version=datalabs.Version("1.0.0"),
            description="SQuALITY dataset for summarization, v1.1 summarization version",
            task_templates=[get_task(TaskType.multi_ref_query_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column=_KEY)]
        )
    ]
    DEFAULT_CONFIG_NAME = "v1"

    def _info(self):

        # Should return a datalab.DatasetInfo object
        features_sample = datalabs.Features(
            {
                _ARTICLE: datalabs.Value("string"),
                _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                _KEY: datalabs.Value("string"),
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._URL[self.config.name]["train"])
        val_path = dl_manager.download(self._URL[self.config.name]["val"])
        test_path = dl_manager.download(self._URL[self.config.name]["test"])
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate SQuALITY examples."""
        _id = 0
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                src = data["document"]
                for x in data["questions"]:
                    query = x["question_text"]
                    summaries = [r["response_text"] for r in x["responses"]]
                    yield _id, {_ARTICLE: src, _ABSTRACT: summaries, _KEY: query}
                    _id += 1
        