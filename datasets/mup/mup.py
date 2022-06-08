import json
import os
import datalabs
from datalabs import get_task, TaskType
import csv
import sys

csv.field_size_limit(sys.maxsize)

_DESCRIPTION = """
 We introduce a novel summarization corpus, leveraging data from scientific peer reviews to capture diverse perspectives from the reader's point of view.
 We leverage data from OpenReview, an open and publicly available platform for scientific publishing. 
 We collect a corpus of papers and their reviews from venues on openreview such as ICLR, NeurIPS, and AKBC primarily from the AI, Machine Learning and Natural Language Processing fields. 
 We use carefully designed heuristics to only include first paragraphs of reviews that are summary-like. 
 We manually check the summaries obtained from this approach on a subset of the data and ensure the high quality of the summaries. 
 The corpus contains a total of around 10K papers, and 26.5K summaries (with average number of 2.57 summaries per paper). 
 The summaries are on average 100.1 words long (space tokenized).
"""

_CITATION = None

_ABSTRACT = "summaries"
_ARTICLE = "text"
_HOMEPAGE = "https://github.com/allenai/mup"
_LICENSE = "ODC-BY"

class MuPConfig(datalabs.BuilderConfig):
    """BuilderConfig for MuP."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MuP.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MuPConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MuPDataset(datalabs.GeneratorBasedBuilder):
    """MuP Dataset."""
    _URL = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/mup-dataset/mup.zip"
    BUILDER_CONFIGS = [
        MuPConfig(
            name="mup",
            version=datalabs.Version("1.0.0"),
            description="MuP dataset for summarization",
            task_templates=[get_task(TaskType.multi_ref_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "mup"

    def _info(self):

        # Should return a datalab.DatasetInfo object
        features_sample = datalabs.Features(
            {
                _ARTICLE: datalabs.Value("string"),
                _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
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
        path = dl_manager.download_and_extract(self._URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": os.path.join(path, "training.csv")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": os.path.join(path, "validation.csv")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate MuP examples."""
        data = dict()
        with open(f_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = row["paper_id"]
                if id not in data:
                    data[id] = dict()
                    data[id]["text"] = row["text"]
                    data[id]["abstract"] = list()
                data[id]["abstract"].append(row["summary"])
        _id = 0
        for id in data:
            yield _id, {
                _ARTICLE: data[id]["text"],
                _ABSTRACT: data[id]["abstract"],
            }
            _id += 1