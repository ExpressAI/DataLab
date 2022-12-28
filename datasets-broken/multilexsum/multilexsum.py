"""Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities"""
import os
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{shen2022multi,
  title={Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities},
  author={Shen, Zejiang and Lo, Kyle and Yu, Lauren and Dahlberg, Nathan and Schlanger, Margo and Downey, Doug},
  journal={arXiv preprint arXiv:2206.10883},
  year={2022}
}
"""

_DESCRIPTION = """\
Multi-LexSum is a multi-doc summarization dataset for civil rights litigation lawsuits with summaries of three granularities.
see: https://arxiv.org/pdf/2206.10883.pdf
"""

_HOMEPAGE = "https://github.com/multilexsum/dataset"
_LICENSE = "Open Data Commons Attribution License (ODC-By)"
_ARTICLE = "texts"
_ABSTRACT = "summary"


class MultiLexSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for MultiLexSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MultiLexSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiLexSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MultiLexSumDataset(datalabs.GeneratorBasedBuilder):
    """MultiLexSum Dataset."""

    BUILDER_CONFIGS = [
        MultiLexSumConfig(
            name="long",
            version=datalabs.Version("1.0.0"),
            description="Multi-doc summarization dataset for civil rights litigation lawsuits.",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        MultiLexSumConfig(
            name="short",
            version=datalabs.Version("1.0.0"),
            description="Multi-doc summarization dataset for civil rights litigation lawsuits.",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        MultiLexSumConfig(
            name="tiny",
            version=datalabs.Version("1.0.0"),
            description="Multi-doc summarization dataset for civil rights litigation lawsuits.",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "long"

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
            license=_LICENSE,
            citation=_CITATION,
            version=self.VERSION,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        train_url = "https://ai2-s2-research.s3.us-west-2.amazonaws.com/multilexsum/releases/v20220616/train.json"
        dev_url = "https://ai2-s2-research.s3.us-west-2.amazonaws.com/multilexsum/releases/v20220616/dev.json"
        test_url = "https://ai2-s2-research.s3.us-west-2.amazonaws.com/multilexsum/releases/v20220616/test.json"
        source_url = "https://ai2-s2-research.s3.us-west-2.amazonaws.com/multilexsum/releases/v20220616/sources.json"

        train_f_path = dl_manager.download(train_url)
        valid_f_path = dl_manager.download(dev_url)
        test_f_path = dl_manager.download(test_url)
        source_f_path = dl_manager.download(source_url)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path, "source_f_path": source_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": valid_f_path, "source_f_path": source_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path, "source_f_path": source_f_path}
            )
        ]

    def _generate_examples(self, f_path, source_f_path):
        """Generate MultiLex examples."""

        with open(source_f_path, "r") as fp:
            source_data = json.load(fp)

        with open(f_path, "r") as fp:
            jsonl_content = fp.read()

        original_datas = [json.loads(jline) for jline in jsonl_content.splitlines()]

        for id_, original_data in enumerate(original_datas):
            source_docs = [
                source_data[source_id]["doc_text"]
                for source_id in original_data["case_documents"]
            ]
            summary = original_data["summary/{}".format(self.config.name)].replace("\n", "").strip()

            raw_feature_info = {
                _ARTICLE: source_docs,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
