"""PTS: Product Title Summarization Corpus."""
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{Sun2018MultiSourcePN,
  title={Multi-Source Pointer Network for Product Title Summarization},
  author={Fei Sun and Peng Jiang and Hanxiao Sun and Changhua Pei and Wenwu Ou and Xiaobo Wang},
  journal={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
  year={2018}
}
"""

_DESCRIPTION = """\
PTS is a new product title summarization dataset from Taobao.com.
Eventually, we get a dataset with 411,267 pairs in 94 categories.
We randomly stratified split the dataset into a training set (80%, 329,248
pairs), a validation set (10%, 41,031 pairs), and a test set (10%, 40,988
pairs) by preserving the percentage of samples for each category.
see: https://arxiv.org/pdf/1808.06885.pdf
"""

_HOMEPAGE = "https://github.com/FeiSun/ProductTitleSummarizationCorpus"
_ABSTRACT = "summary"
_ARTICLE = "text"


class PTSConfig(datalabs.BuilderConfig):
    """BuilderConfig for PTS."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for PTS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PTSConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class PTSDataset(datalabs.GeneratorBasedBuilder):
    """PTS Dataset."""

    BUILDER_CONFIGS = [
        PTSConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Product title summarization corpus.",
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
            languages=["zh"],
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )

    def _split_generators(self, dl_manager):
        train_f_path = dl_manager.download(
            "https://raw.githubusercontent.com/FeiSun/ProductTitleSummarizationCorpus/master/corpus/train.txt")
        valid_f_path = dl_manager.download(
            "https://raw.githubusercontent.com/FeiSun/ProductTitleSummarizationCorpus/master/corpus/val.txt")
        test_f_path = dl_manager.download(
            "https://raw.githubusercontent.com/FeiSun/ProductTitleSummarizationCorpus/master/corpus/test.txt")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate PTS examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            article = line.split("\t\t")[0].strip()
            summary = line.split("\t\t")[1].strip()
            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: article,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
