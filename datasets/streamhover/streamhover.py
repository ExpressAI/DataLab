"""StreamHover: Livestream Transcript Summarization and Annotation"""
import os
import pickle

import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{Cho2021StreamHoverLT,
  title={StreamHover: Livestream Transcript Summarization and Annotation},
  author={Sangwoo Cho and Franck Dernoncourt and Tim Ganter and Trung Bui and Nedim Lipka and Walter Chang and Hailin Jin and Jonathan Brandt and Hassan Foroosh and Fei Liu},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.05160}
}
"""

_DESCRIPTION = """\
We create a new benchmark dataset for automatic summarization of livestream transcripts.
see: https://arxiv.org/pdf/2109.05160.pdf
"""

_HOMEPAGE = "https://github.com/ucfnlp/streamhover"
_ARTICLE = "text"
_ABSTRACT = "summary"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class StreamHoverConfig(datalabs.BuilderConfig):
    """BuilderConfig for StreamHover."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for StreamHover.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(StreamHoverConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class StreamHoverDataset(datalabs.GeneratorBasedBuilder):
    """StreamHover Dataset."""

    _FILE_ID = "1kMmMX7ceYLOZuhdsgi_Qahc269Bpipha"

    BUILDER_CONFIGS = [
        StreamHoverConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Livestream transcript summarization dataset.",
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
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "Behance_train.pkl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": os.path.join(f_path, "Behance_val.pkl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": os.path.join(f_path, "Behance_test.pkl")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate StreamHover examples."""
        with open(f_path, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        datas = []
        for data in dataset:
            transcripts = [transcript["display"].strip() for transcript in data[0]]
            text = " ".join(transcripts)
            summary = data[1].strip()
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {_ARTICLE: text, _ABSTRACT: summary}
            yield id_, raw_feature_info
