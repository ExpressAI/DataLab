import json
import os
import subprocess
import tempfile

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
)

import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.more_features import get_feature_schemas

_DESCRIPTION = """
 GovReport dataset for summarization.
 From paper: "Efficient Attentions for Long Document Summarization" by L. Huang et al.
 See: https://aclanthology.org/2021.naacl-main.112.pdf
 See: https://github.com/luyang-huang96/LongDocSum
"""
_CITATION = """\
    @inproceedings{huang-etal-2021-efficient,
    title = "Efficient Attentions for Long Document Summarization",
    author = "Huang, Luyang  and
      Cao, Shuyang  and
      Parulian, Nikolaus  and
      Ji, Heng  and
      Wang, Lu",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.112",
    doi = "10.18653/v1/2021.naacl-main.112",
    pages = "1419--1436",
    abstract = "The quadratic computational and memory complexities of large Transformers have limited their scalability for long document summarization. In this paper, we propose Hepos, a novel efficient encoder-decoder attention with head-wise positional strides to effectively pinpoint salient information from the source. We further conduct a systematic study of existing efficient self-attentions. Combined with Hepos, we are able to process ten times more tokens than existing models that use full attentions. For evaluation, we present a new dataset, GovReport, with significantly longer documents and summaries. Results show that our models produce significantly higher ROUGE scores than competitive comparisons, including new state-of-the-art results on PubMed. Human evaluation also shows that our models generate more informative summaries with fewer unfaithful errors.",
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


class GovReportConfig(datalabs.BuilderConfig):
    """BuilderConfig for GovReport."""

    def __init__(self, **kwargs):
        """BuilderConfig for GovReport.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GovReportConfig, self).__init__(**kwargs)


class GovReportDataset(datalabs.GeneratorBasedBuilder):
    """GovReport Dataset."""

    # _FILE = "https://drive.google.com/uc?id=1AwaWVVlv77gbWXwMzUX46r4kq2yY4Ylh&export=download"
    _FILE = _gdrive_url("1AwaWVVlv77gbWXwMzUX46r4kq2yY4Ylh")
    BUILDER_CONFIGS = [
        GovReportConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="GovReport dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):

        features_dataset = {}
        features_sample = datalabs.Features(
            {
                _ARTICLE: datalabs.Value("string"),
                _ABSTRACT: datalabs.Value("string"),
                # "id": datalab.Value("string"),
            }
        )
        if self.feature_expanding:
            features_sample, features_dataset = get_feature_schemas(
                features_sample, get_schema_of_sample_level_features
            )

        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage="https://github.com/luyang-huang96/LongDocSum",
            citation=_CITATION,
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                ),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_custom(self._FILE, custom_download)
        f_path = dl_manager.extract(f_path)
        train_src_path = os.path.join(f_path, "gov_report_fairseq_format/train.source")
        train_tgt_path = os.path.join(f_path, "gov_report_fairseq_format/train.target")
        val_src_path = os.path.join(f_path, "gov_report_fairseq_format/val.source")
        val_tgt_path = os.path.join(f_path, "gov_report_fairseq_format/val.target")
        test_src_path = os.path.join(f_path, "gov_report_fairseq_format/test.source")
        test_tgt_path = os.path.join(f_path, "gov_report_fairseq_format/test.target")

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
        """Generate GovReport examples."""
        with open(src_path, encoding="utf-8") as f_src, open(
            tgt_path, encoding="utf-8"
        ) as f_tgt:
            for (id_, (row_src, row_tgt)) in enumerate(zip(f_src, f_tgt)):

                # if id_ > 20:
                #     break

                row_src = row_src.strip()
                row_tgt = row_tgt.strip()

                raw_feature_info = {"text": row_src, "summary": row_tgt}

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(
                        raw_feature_info
                    )
                    raw_feature_info.update(additional_feature_info)
                    # print(additional_feature_info)
                    yield id_, raw_feature_info
