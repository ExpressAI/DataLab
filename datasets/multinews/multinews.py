import json
from operator import imod
import os
import subprocess
import tempfile

import datalabs
from datalabs import get_task, TaskType
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN

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


def custom_download(url, path):
    with tempfile.TemporaryDirectory() as tmpdir:
        response = subprocess.check_output(
            [
                "wget",
                "--save-cookies",
                os.path.join(tmpdir, "cookies.txt"),
                f"{url}",
                "-O-",
            ]
        )
        with open(os.path.join(tmpdir, "response.txt"), "w") as f:
            f.write(response.decode("utf-8"))
        response = subprocess.check_output(
            [
                "sed",
                "-rn",
                "s/.*confirm=([0-9A-Za-z_]+).*/\\1/p",
                os.path.join(tmpdir, "response.txt"),
            ]
        )
        response = response.decode("utf-8")
        subprocess.check_output(
            [
                "wget",
                "--load-cookies",
                os.path.join(tmpdir, "cookies.txt"),
                "-O",
                path,
                url + f"&confirm={response}",
            ]
        )


class MultiNewsConfig(datalabs.BuilderConfig):
    """BuilderConfig for MultiNews."""

    def __init__(self, name, version, description, task_templates, **kwargs):
        """BuilderConfig for MultiNews.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiNewsConfig, self).__init__(**kwargs)
        self.name = name
        self.version = version
        self.description = description
        self.task_templates = task_templates


class MultiNewsDataset(datalabs.GeneratorBasedBuilder):
    """MultiNews Dataset."""

    BUILDER_CONFIGS = [
        MultiNewsConfig(
            name="raw-single",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, single document version, with raw data",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="raw-cleaned-single",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, single document version, with cleaned raw data, see issue https://github.com/Alex-Fabbri/Multi-News/issues/11",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="preprocessed-single",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, single document version, with preprocessed data",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="truncated-single",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, single document version, with preprocessed and truncated data",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="raw-multi",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, multi-document version, with raw data",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="raw-cleaned-multi",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, multi-document version, with cleaned raw data, see issue https://github.com/Alex-Fabbri/Multi-News/issues/11",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="preprocessed-multi",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, multi-document version, with preprocessed data",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
        MultiNewsConfig(
            name="truncated-multi",
            version=datalabs.Version("1.0.0"),
            description="MultiNews dataset for summarization, multi-document version, with preprocessed and truncated data",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "raw-multi"

    def _info(self):
        # Should return a datalab.DatasetInfo object

        if "multi" in self.config.name:
            features_sample = datalabs.Features(
                {
                    _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        else:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage="https://github.com/Alex-Fabbri/Multi-News",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        if self.config.name in ["raw-single", "raw-multi"]:
            train_src_path = dl_manager.download_custom(
                _gdrive_url("1vWfhWIj-UpV_bY-zcu7lw4m9z8hLkF0r"), custom_download
            )
            train_tgt_path = dl_manager.download_custom(
                _gdrive_url("1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq"), custom_download
            )
            val_src_path = dl_manager.download_custom(
                _gdrive_url("1L2dk4ThZ-Bau9rIQpMG8I75R15FpLE-B"), custom_download
            )
            val_tgt_path = dl_manager.download_custom(
                _gdrive_url("1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM"), custom_download
            )
            test_src_path = dl_manager.download_custom(
                _gdrive_url("1_jyJOVkAfRafJQkH2HLYhw4NTKU5f4bq"), custom_download
            )
            test_tgt_path = dl_manager.download_custom(
                _gdrive_url("1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp"), custom_download
            )
        elif self.config.name in ["raw-cleaned-single", "raw-cleaned-multi"]:
            train_src_path = dl_manager.download_custom(
                _gdrive_url("1wHAWDOwOoQWSj7HYpyJ3Aeud8WhhaJ7P"), custom_download
            )
            val_src_path = dl_manager.download_custom(
                _gdrive_url("1p_u9_jpz3Zbj0EL05QFX6wvJAahmOn6h"), custom_download
            )
            test_src_path = dl_manager.download_custom(
                _gdrive_url("1-n_6fj-1nM7sWtBSNkQCSfl5Rb3zPVfr"), custom_download
            )
            train_tgt_path = dl_manager.download_custom(
                _gdrive_url("1QVgswwhVTkd3VLCzajK6eVkcrSWEK6kq"), custom_download
            )
            val_tgt_path = dl_manager.download_custom(
                _gdrive_url("1Y1lBbBU5Q0aJMqLhYEOdEtTqQ85XnRRM"), custom_download
            )
            test_tgt_path = dl_manager.download_custom(
                _gdrive_url("1CX_YcgQ3WwNC1fXBpMfwMXFPCqsd9Lbp"), custom_download
            )
        elif self.config.name == ["preprocessed-single", "preprocessed-multi"]:
            train_src_path = dl_manager.download_custom(
                _gdrive_url("166MtnlB8eEGpH6UZLKgGNsk9u6EDdQ8E"), custom_download
            )
            train_tgt_path = dl_manager.download_custom(
                _gdrive_url("1JniyQbgWdiS-tnDEweTlQxkFE9lRsQJU"), custom_download
            )
            val_src_path = dl_manager.download_custom(
                _gdrive_url("1RzmVVqVMNWhjNTUWKeiBS-HW1UIqnXeS"), custom_download
            )
            val_tgt_path = dl_manager.download_custom(
                _gdrive_url("1fpLqEb4lQ2F0ooBzyBoVc-d2S1qh-euS"), custom_download
            )
            test_src_path = dl_manager.download_custom(
                _gdrive_url("1trAjuswWLs57rgJaC7ZQFFNik8-p77Qf"), custom_download
            )
            test_tgt_path = dl_manager.download_custom(
                _gdrive_url("1JTPHdYYEMm9-VFNWuDD2hGJARO-3fyXI"), custom_download
            )
        else:
            train_src_path = dl_manager.download_custom(
                _gdrive_url("17x4TH2NRHyP4EJGPaX0P3P5sFhrOKMyP"), custom_download
            )
            train_tgt_path = dl_manager.download_custom(
                _gdrive_url("1WNB0JGAHUS6Fl2-ZZERtq_MhKVR06EL4"), custom_download
            )
            val_src_path = dl_manager.download_custom(
                _gdrive_url("1YXkF_ugMx1HYCBBYF7VKq0Lujzif1JES"), custom_download
            )
            val_tgt_path = dl_manager.download_custom(
                _gdrive_url("11C3k3XW1MpQEKymftQPsqSxcGEq_M7Xj"), custom_download
            )
            test_src_path = dl_manager.download_custom(
                _gdrive_url("1-UnukKI0rRfxpEwCHUXykGlxj7UllEiD"), custom_download
            )
            test_tgt_path = dl_manager.download_custom(
                _gdrive_url("1YDjw1yPwgN-mqzqwWzbxKBIRkpnaJQDY"), custom_download
            )

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"src_path": train_src_path, "tgt_path": train_tgt_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"src_path": val_src_path, "tgt_path": val_tgt_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"src_path": test_src_path, "tgt_path": test_tgt_path},
            ),
        ]

    def _generate_examples(self, src_path, tgt_path):
        """Generate MultiNews examples."""
        if self.config.name in ["raw-single", "raw-cleaned-single"]:
            with open(src_path, encoding="utf-8") as f_src, open(
                tgt_path, encoding="utf-8"
            ) as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = row_src.strip().replace("NEWLINE_CHAR", "")
                    row_tgt = row_tgt.strip().lstrip("– ")

                    raw_feature_info = {"text": row_src, "summary": row_tgt}
                    yield id_, raw_feature_info

        elif self.config.name in ["preprocessed-single", "truncated-single"]:
            with open(src_path, encoding="utf-8") as f_src, open(
                tgt_path, encoding="utf-8"
            ) as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = row_src.strip().replace(
                        "story_separator_special_tag", "|||||"
                    )
                    row_tgt = row_tgt.strip().lstrip("– ")

                    raw_feature_info = {"text": row_src, "summary": row_tgt}

                    yield id_, raw_feature_info

        elif self.config.name in ["raw-multi", "raw-cleaned-multi"]:
            with open(src_path, encoding="utf-8") as f_src, open(
                tgt_path, encoding="utf-8"
            ) as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = row_src.strip().replace("NEWLINE_CHAR", "")
                    row_src = [x.strip() for x in row_src.split("|||||")]
                    row_src = [x for x in row_src if len(x) > 0]
                    row_tgt = row_tgt.strip().lstrip("– ")
                    yield id_, {_MDS_TEXT_COLUMN: row_src, "summary": row_tgt}
        else:
            with open(src_path, encoding="utf-8") as f_src, open(
                tgt_path, encoding="utf-8"
            ) as f_tgt:
                for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):
                    row_src = [
                        x.strip()
                        for x in row_src.strip().split("story_separator_special_tag")
                    ]
                    row_src = [x for x in row_src if len(x) > 0]
                    row_tgt = row_tgt.strip().lstrip("– ")
                    yield id_, {_MDS_TEXT_COLUMN: row_src, "summary": row_tgt}
