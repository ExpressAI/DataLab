import json
import os
import datalabs
from datalabs import get_task, TaskType
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN
import csv
import sys

csv.field_size_limit(sys.maxsize)


_DESCRIPTION = """
 This is a dataset of 4.5K reviews collected from Cochrane systematic reviews. These are reviews of all trials relevant to a given clinical question. 
 The systematic review abstracts together with the titles and abstracts of the clinical trials summarized by these reviews form our dataset. 
 All data was downloaded via PubMed (i.e., we use only abstracts). 
"""

_CITATION = """\
    @article{Wallace2020GeneratingN,
    title={Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization},
    author={Byron C. Wallace and Sayantani Saha and Frank Soboczenski and Iain James Marshall},
    journal={AMIA Annual Symposium},
    year={2020},
    volume={abs/2008.11293}
    }
"""

_ABSTRACT = "summary"
_ARTICLE = "text"


class CochraneConfig(datalabs.BuilderConfig):
    """BuilderConfig for Cochrane."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Cochrane.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CochraneConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class CochraneDataset(datalabs.GeneratorBasedBuilder):
    """Cochrane Dataset."""
    _URL = "https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"
    BUILDER_CONFIGS = [
        CochraneConfig(
            name="cochrane",
            version=datalabs.Version("1.0.0"),
            description="Cochrane dataset for multi-document summarization",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_MDS_TEXT_COLUMN,
                reference_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "cochrane"

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
            homepage="https://github.com/allenai/mslr-shared-task",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download_and_extract(self._URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/cochrane/", "train-inputs.csv"),
                    "tgt_path": os.path.join(path, "mslr_data/cochrane/", "train-targets.csv"),
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/cochrane/", "dev-inputs.csv"),
                    "tgt_path": os.path.join(path, "mslr_data/cochrane/", "dev-targets.csv"),
                }
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={
                    "src_path": os.path.join(path, "mslr_data/cochrane/", "test-inputs.csv"),
                    "tgt_path": None,
                }
            ),
        ]

    def _generate_examples(self, src_path, tgt_path):
        """Generate Cochrane examples."""
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
        
