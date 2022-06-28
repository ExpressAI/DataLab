import json
import os
import datalabs
from datalabs import get_task, TaskType
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN
import csv
import sys

csv.field_size_limit(sys.maxsize)


_DESCRIPTION = """
 MS^2 is a dataset containing medical systematic reviews, their constituent studies, and a large amount of related markup. 
 This dataset is created as an annotated subset of the Semantic Scholar research corpus.
"""

_CITATION = """\
    @inproceedings{DeYoung2021MS2MS,
      title={MSˆ2: Multi-Document Summarization of Medical Studies},
      author={Jay DeYoung and Iz Beltagy and Madeleine van Zuylen and Bailey Kuehl and Lucy Lu Wang},
      booktitle={EMNLP},
      year={2021}
    }
"""

_ABSTRACT = "summary"
_ARTICLE = "text"


class MS2Config(datalabs.BuilderConfig):
    """BuilderConfig for MS2."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MS2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MS2Config, self).__init__(**kwargs)
        self.task_templates = task_templates


class MS2Dataset(datalabs.GeneratorBasedBuilder):
    """MS2 Dataset."""
    _URL = "https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"
    BUILDER_CONFIGS = [
        MS2Config(
            name="MS2",
            version=datalabs.Version("1.0.0"),
            description="MS2 dataset for multi-document summarization",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_MDS_TEXT_COLUMN,
                reference_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "MS2"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_sample = datalabs.Features(
            {
                _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                _ABSTRACT: datalabs.Value("string"),
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage="https://github.com/allenai/ms2",
            citation=_CITATION,
            task_templates=self.config.task_templates,
            license="Semantic Scholar API and Dataset License Agreement"
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(self._URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/ms2/", "train-inputs.csv"),
                    "tgt_path": os.path.join(path, "mslr_data/ms2/", "train-targets.csv"),
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/ms2/", "dev-inputs.csv"),
                    "tgt_path": os.path.join(path, "mslr_data/ms2/", "dev-targets.csv"),
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/ms2/", "test-inputs.csv"),
                    "tgt_path": None,
                }
            ),
        ]

    def _generate_examples(self, src_path, tgt_path):
        """Generate MS2 examples."""
        data = dict()
        with open(src_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = row["ReviewID"]
                if id not in data:
                    data[id] = []
                data[id].append(row["Abstract"])
        if tgt_path is not None:
            with open(tgt_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                _id = 0
                for row in reader:
                    id = row["ReviewID"]
                    if id in data:
                        yield _id, {
                            _MDS_TEXT_COLUMN: data[id],
                            _ABSTRACT: row["Target"],
                        }
                        _id += 1
        else:
            for _id, row in enumerate(data.values()):
                yield _id, {
                    _MDS_TEXT_COLUMN: row,
                    _ABSTRACT: None,
                }
        
