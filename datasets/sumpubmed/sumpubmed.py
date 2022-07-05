"""SumPubMed: Summarization Dataset of PubMed Scientific Articles"""
import os
import glob
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{gupta-etal-2021-sumpubmed,
    title = "{SumPubMed}: Summarization Dataset of {P}ub{M}ed Scientific Articles",
    author = "Gupta, Vivek  and
      Bharti, Prerna  and
      Nokhiz, Pegah  and
      Karnick, Harish",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-srw.30",
    doi = "10.18653/v1/2021.acl-srw.30",
    pages = "292--303"
}
"""

_DESCRIPTION = """\
SUMPUBMED is a dataset for abstractive summarization over scientific article. 
The dataset use the PubMed bio-medical article to create the SUMPUBMED summarization dataset. 
PubMed comprises of more than 26 million citations for biomedical literature from MEDLINE, 
life science journals, and online books. Citations may include links to full-text content 
from PubMed Central and publisher web sites.
see: https://aclanthology.org/2021.acl-srw.30.pdf
"""

_HOMEPAGE = "https://github.com/vgupta123/sumpubmed"
_LICENSE = "MIT License"
_ARTICLE = "text"
_ABSTRACT = "summary"


class SumPubMedConfig(datalabs.BuilderConfig):
    """BuilderConfig for SumPubMed."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for SumPubMed.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SumPubMedConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class SumPubMedDataset(datalabs.GeneratorBasedBuilder):
    """SumPubMed Dataset."""

    BUILDER_CONFIGS = [
        SumPubMedConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="SUMPUBMED is a dataset for abstractive summarization over scientific article.",
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
        url = "https://github.com/vgupta123/sumpubmed/archive/refs/heads/master.zip"
        f_path = dl_manager.download_and_extract(url)
        f_path = os.path.join(f_path, "sumpubmed-master")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": f_path, "split": "train"}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": f_path, "split": "valid"}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": f_path, "split": "test"}
            ),
        ]

    def _generate_examples(self, f_path, split):
        """Generate SumPubMed examples."""
        text_files = glob.glob(os.path.join(f_path, "line_text", "*.txt"))
        datas = []
        for text_file in text_files:
            file_name = os.path.basename(text_file)
            file_id = file_name.replace(".txt", "").split("_")[1]
            summary_file = os.path.join(f_path, "shorter_abstract", "abst_{}.txt".format(file_id))
            summary_f = open(summary_file, encoding="utf-8")
            summary_lines = summary_f.readlines()
            summary = " ".join([summary_line.strip() for summary_line in summary_lines])

            text_f = open(text_file, encoding="utf-8")
            text_lines = text_f.readlines()
            text = " ".join([text_line.strip() for text_line in text_lines])
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
