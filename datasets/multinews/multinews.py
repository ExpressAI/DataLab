import json
import os
import datalabs
from datalabs.tasks import Summarization

_DESCRIPTION = """
 Multinews dataset for mutlti-document summarization.
 Each data sample contains multuple documents, which are seperated by "|||||"
 From paper: "Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model" by A. Fabbri et al.
 See: https://aclanthology.org/P19-1102.pdf
 See: https://github.com/Alex-Fabbri/Multi-News
"""
_CITATION = """\
    @inproceedings{fabbri-etal-2019-multi,
    title = "Multi-News: A Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model",
    author = "Fabbri, Alexander  and
      Li, Irene  and
      She, Tianwei  and
      Li, Suyi  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1102",
    doi = "10.18653/v1/P19-1102",
    pages = "1074--1084",
    abstract = "Automatic generation of summaries from multiple news articles is a valuable tool as the number of online publications grows rapidly. Single document summarization (SDS) systems have benefited from advances in neural encoder-decoder model thanks to the availability of large datasets. However, multi-document summarization (MDS) of news articles has been limited to datasets of a couple of hundred examples. In this paper, we introduce Multi-News, the first large-scale MDS news dataset. Additionally, we propose an end-to-end model which incorporates a traditional extractive summarization model with a standard SDS model and achieves competitive results on MDS datasets. We benchmark several methods on Multi-News and hope that this work will promote advances in summarization in the multi-document setting.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"

class MultiNewsConfig(datalabs.BuilderConfig):
    """BuilderConfig for MultiNews."""

    def __init__(self, **kwargs):
        """BuilderConfig for MultiNews.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiNewsConfig, self).__init__(**kwargs)


class MultiNewsDataset(datalabs.GeneratorBasedBuilder):
    """MultiNews Dataset."""
    BUILDER_CONFIGS = [
        MultiNewsConfig(
            name="raw",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, with raw data",
        ),
        MultiNewsConfig(
            name="raw-cleaned",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, with cleaned raw data, see issue https://github.com/Alex-Fabbri/Multi-News/issues/11",
        ),
        MultiNewsConfig(
            name="preprocessed",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, with preprocessed data",
        ),
        MultiNewsConfig(
            name="truncated",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, with preprocessed and truncated data",
        ),
        
    ]
    DEFAULT_CONFIG_NAME = "raw"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    # "id": datalab.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/Alex-Fabbri/Multi-News",
            citation=_CITATION,
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "raw":
            train_src_path = dl_manager.download(_gdrive_url("1vWfhWIj-UpV_bY-zcu7lw4m9z8hLkF0r"))
            train_tgt_path = dl_manager.download(_gdrive_url("1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq"))
            val_src_path = dl_manager.download(_gdrive_url("1L2dk4ThZ-Bau9rIQpMG8I75R15FpLE-B"))
            val_tgt_path = dl_manager.download(_gdrive_url("1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM"))
            test_src_path = dl_manager.download(_gdrive_url("1_jyJOVkAfRafJQkH2HLYhw4NTKU5f4bq"))
            test_tgt_path = dl_manager.download(_gdrive_url("1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp"))
        elif self.config.name == "raw-cleaned":
            train_src_path = dl_manager.download(_gdrive_url("1wHAWDOwOoQWSj7HYpyJ3Aeud8WhhaJ7P"))
            val_src_path = dl_manager.download(_gdrive_url("1p_u9_jpz3Zbj0EL05QFX6wvJAahmOn6h"))
            test_src_path = dl_manager.download(_gdrive_url("1-n_6fj-1nM7sWtBSNkQCSfl5Rb3zPVfr"))
            train_tgt_path = dl_manager.download(_gdrive_url("1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq"))
            val_tgt_path = dl_manager.download(_gdrive_url("1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM"))
            test_tgt_path = dl_manager.download(_gdrive_url("1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp"))
        elif self.config.name == "preprocessed":
            train_src_path = dl_manager.download(_gdrive_url("166MtnlB8eEGpH6UZLKgGNsk9u6EDdQ8E"))
            train_tgt_path = dl_manager.download(_gdrive_url("1JniyQbgWdiS-tnDEweTlQxkFE9lRsQJU"))
            val_src_path = dl_manager.download(_gdrive_url("1RzmVVqVMNWhjNTUWKeiBS-HW1UIqnXeS"))
            val_tgt_path = dl_manager.download(_gdrive_url("1fpLqEb4lQ2F0ooBzyBoVc-d2S1qh-euS"))
            test_src_path = dl_manager.download(_gdrive_url("1trAjuswWLs57rgJaC7ZQFFNik8-p77Qf"))
            test_tgt_path = dl_manager.download(_gdrive_url("1JTPHdYYEMm9-VFNWuDD2hGJARO-3fyXI"))
        else:
            train_src_path = dl_manager.download(_gdrive_url("17x4TH2NRHyP4EJGPaX0P3P5sFhrOKMyP"))
            train_tgt_path = dl_manager.download(_gdrive_url("1WNB0JGAHUS6Fl2-ZZERtq_MhKVR06EL4"))
            val_src_path = dl_manager.download(_gdrive_url("1YXkF_ugMx1HYCBBYF7VKq0Lujzif1JES"))
            val_tgt_path = dl_manager.download(_gdrive_url("11C3k3XW1MpQEKymftQPsqSxcGEq_M7Xj"))
            test_src_path = dl_manager.download(_gdrive_url("1-UnukKI0rRfxpEwCHUXykGlxj7UllEiD"))
            test_tgt_path = dl_manager.download(_gdrive_url("1YDjw1yPwgN-mqzqwWzbxKBIRkpnaJQDY"))
        
        
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
        """Generate MultiNews examples."""
        if self.config.name in ["raw", "raw-cleaned"]:
            with open(src_path, encoding="utf-8") as f_src, open(tgt_path, encoding="utf-8") as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = row_src.strip().replace("NEWLINE_CHAR", ""),
                    row_tgt = row_tgt.strip().lstrip("- ")
                    yield id_, {"text": row_src, "summary": row_tgt}
        else:
            with open(src_path, encoding="utf-8") as f_src, open(tgt_path, encoding="utf-8") as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = row_src.strip().replace("story_separator_special_tag", "|||||"),
                    row_tgt = row_tgt.strip().lstrip("- ")
                    yield id_, {"text": row_src, "summary": row_tgt}
        