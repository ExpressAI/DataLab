"""pn_summary: Persian Abstractive Text Summarization Dataset."""
import os
import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@article{pnSummary,
  title={Leveraging ParsBERT and Pretrained mT5 for Persian Abstractive Text Summarization},
  author={Mehrdad Farahani and Mohammad Gharachorloo and M. Manthouri},
  journal={2021 26th International Computer Conference, Computer Society of Iran (CSICC)},
  year={2021},
  pages={1-6},
  doi={10.1109/CSICC52343.2021.9420563},
}
"""

_DESCRIPTION = """\
A well-structured summarization dataset for the Persian language consists of 93,207 records. 
It is prepared for Abstractive/Extractive tasks (like cnn_dailymail for English). 
It can also be used in other scopes like Text Generation, Title Generation, 
and News Category Classification. Moreover, we tested out this dataset on novel models and techniques.
see: https://arxiv.org/pdf/2012.11204.pdf
"""

_HOMEPAGE = "https://github.com/hooshvare/pn-summary"
_LICENSE = "MIT License"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class PnSummaryConfig(datalabs.BuilderConfig):
    """BuilderConfig for pn_summary."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for pn_summary.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PnSummaryConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class PnSummaryDataset(datalabs.GeneratorBasedBuilder):
    """PnSummary Dataset."""
    _FILE_ID = "16OgJ_OrfzUF_i3ftLjFn9kpcyoi7UJeO"

    BUILDER_CONFIGS = [
        PnSummaryConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Persian Abstractive Text Summarization Dataset.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
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
            license=_LICENSE,
            version=self.VERSION,
            languages=["fa"],  # https://huggingface.co/languages#:~:text=57-,Persian,-fa
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "pn_summary/train.csv")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "pn_summary/dev.csv")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(f_path, "pn_summary/test.csv")}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate pn_summary examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()[1:]  # remove the csv header
        datas = []
        # simple way to process csv files
        for line in lines:
            article = line.split("\t")[2].strip()
            summary = line.split("\t")[3].strip()
            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: article,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
