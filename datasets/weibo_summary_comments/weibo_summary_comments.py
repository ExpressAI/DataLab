"""weibo_summary_comments: Abstractive Text Summarization by Incorporating Reader Comments"""
import os
import csv
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{Gao2019AbstractiveTS,
  title={Abstractive Text Summarization by Incorporating Reader Comments},
  author={Shen Gao and Xiuying Chen and Piji Li and Zhaochun Ren and Lidong Bing and Dongyan Zhao and Rui Yan},
  booktitle={AAAI},
  year={2019}
}
"""

_DESCRIPTION = """\
We collect the document-summary-comments pair data from Weibo which is the largest social network website in
China, and users can read a document and post a comment about the document on this website.
In total, our training dataset contains 863826 training samples.
see: https://arxiv.org/pdf/1812.05407.pdf
"""

_HOMEPAGE = "http://t.cn/EAH5JxS"
_ARTICLE = "text"
_ABSTRACT = "summary"


class WeiboSummaryCommentsConfig(datalabs.BuilderConfig):
    """BuilderConfig for WeiboSummaryComments."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WeiboSummaryComments.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WeiboSummaryCommentsConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WeiboSummaryCommentsDataset(datalabs.GeneratorBasedBuilder):
    """WeiboSummaryComments Dataset."""

    BUILDER_CONFIGS = [
        WeiboSummaryCommentsConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Reader-aware Chinese summarization dataset.",
            task_templates=[get_task(TaskType.reader_aware_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column="comments")]
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    "comments": datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=["zh"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://drive.google.com/u/0/uc?id=1_YH5cBtvNnUNJjGj7kiTMjuHydBqWYQT&export=download&confirm=t"
        f_path = dl_manager.download_and_extract(url)
        f_path = os.path.join(f_path, "merge.json")

        # TODO do not know the split info
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": f_path}
            )
        ]

    def _generate_examples(self, f_path):
        """Generate TWEETSUM examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            original_data = json.loads(line)
            article = original_data["article"].replace(" ", "")
            abstract = original_data["abstract"].replace(" ", "")
            comments = original_data["comments"]
            comments = [comment[0].replace(" ", "") for comment in comments]
            datas.append({"text": article, "comments": comments, "summary": abstract})

        for id_, data in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: data["text"],
                "comments": data["comments"],
                _ABSTRACT: data["summary"]
            }
            yield id_, raw_feature_info
