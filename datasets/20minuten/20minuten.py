"""A New Dataset and Efficient Baselines for Document-level Text Simplification in German"""
import os
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{rios-etal-2021-new,
    title = "A New Dataset and Efficient Baselines for Document-level Text Simplification in {G}erman",
    author = {Rios, Annette  and
      Spring, Nicolas  and
      Kew, Tannon  and
      Kostrzewa, Marek  and
      S{\"a}uberli, Andreas  and
      M{\"u}ller, Mathias  and
      Ebling, Sarah},
    booktitle = "Proceedings of the Third Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2021",
    address = "Online and in Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.newsum-1.16",
    doi = "10.18653/v1/2021.newsum-1.16",
    pages = "152--161"
}
"""

_DESCRIPTION = """\
We introduce a newly collected data set of German texts, collected from the Swiss news magazine 20 Minuten (‘20 Minutes’) that consists of full articles paired with simplified summaries.
see: https://aclanthology.org/2021.newsum-1.16.pdf
"""

_HOMEPAGE = "https://github.com/ZurichNLP/20Minuten"
_ARTICLE = "text"
_ABSTRACT = "summary"


class MinutenConfig(datalabs.BuilderConfig):
    """BuilderConfig for 20 Minuten."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for 20 Minuten.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MinutenConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MinutenDataset(datalabs.GeneratorBasedBuilder):
    """20 Minuten Dataset."""

    BUILDER_CONFIGS = [
        MinutenConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="German document summarization dataset.",
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
            languages=["de"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://20min-simplified-language-uzh.s3.eu-central-1.amazonaws.com/EMNLP_newsum_2021_A_New_Dataset_TS_DE.zip"
        f_path = dl_manager.download_and_extract(url)
        f_path = os.path.join(f_path,
                              "2021_ANewDatasetandEfficientBaselinesforDocument-levelTextSimplificationinGerman/data/dedup")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"src_f_path": os.path.join(f_path, "train.src.no_tag.de"),
                            "tgt_f_path": os.path.join(f_path, "train.trg.no_tag.simpde")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"src_f_path": os.path.join(f_path, "dev.src.no_tag.de"),
                            "tgt_f_path": os.path.join(f_path, "dev.trg.no_tag.simpde")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"src_f_path": os.path.join(f_path, "test.src.no_tag.de"),
                            "tgt_f_path": os.path.join(f_path, "test.trg.no_tag.simpde")}
            )
        ]

    def _generate_examples(self, src_f_path, tgt_f_path):
        """Generate 20Minuten examples."""

        with open(src_f_path, "r") as f_s:
            src_lines = f_s.readlines()

        with open(tgt_f_path, "r") as f_t:
            tgt_lines = f_t.readlines()

        for id_, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            raw_feature_info = {
                _ARTICLE: src_line.strip(),
                _ABSTRACT: tgt_line.strip()
            }
            yield id_, raw_feature_info
