""" Open4Business (O4B): An Open Access Dataset for Summarizing Business Documents """
import json
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@misc{singh2020open4businesso4b,
      title={Open4Business(O4B): An Open Access Dataset for Summarizing Business Documents}, 
      author={Amanpreet Singh and Niranjan Balasubramanian},
      year={2020},
      eprint={2011.07636},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
"""

_DESCRIPTION = """\
Open4Business (O4B), a dataset of 17,458 open access business articles and their reference summaries.
The dataset introduces a new challenge for summarization in the business domain, requiring highly abstractive and more concise summaries as compared to other existing datasets. 
see: https://arxiv.org/pdf/2011.07636.pdf
"""

_HOMEPAGE = "https://github.com/amanpreet692/Open4Business"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"
        

class Open4BusinessConfig(datalabs.BuilderConfig):
    """BuilderConfig for Open4Business."""

    def __init__(self, **kwargs):
        """BuilderConfig for Open4Business.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Open4BusinessConfig, self).__init__(**kwargs)


class Open4BusinessDataset(datalabs.GeneratorBasedBuilder):
    """Open4Business Dataset."""
    _FILE = _gdrive_url("1qJzriJyL6plmdzKdU1HIlR0jUWl-_dxZ")
    BUILDER_CONFIGS = [
        Open4BusinessConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Open4Business dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
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
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._FILE)
        train_src_path = os.path.join(f_path, "Open4Business/train.source")
        train_tgt_path = os.path.join(f_path, "Open4Business/train.target")
        val_src_path = os.path.join(f_path, "Open4Business/val.source")
        val_tgt_path = os.path.join(f_path, "Open4Business/val.target")
        test_src_path = os.path.join(f_path, "Open4Business/test.source")
        test_tgt_path = os.path.join(f_path, "Open4Business/test.target")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"src_path": train_src_path, "tgt_path": train_tgt_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"src_path": val_src_path, "tgt_path": val_tgt_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"src_path": test_src_path, "tgt_path": test_tgt_path}
            ),
        ]

    def _generate_examples(self, src_path, tgt_path):
        """Generate Open4Business examples."""
        with open(src_path, encoding="utf-8") as f_src, open(tgt_path, encoding="utf-8") as f_tgt:
            for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                row_src = row_src.strip()
                row_tgt = row_tgt.strip()
                yield id_, {"text": row_src, "summary": row_tgt}