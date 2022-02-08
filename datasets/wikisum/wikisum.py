import json
import os
import datalabs
from datalabs.tasks import Summarization

_DESCRIPTION = """
 The WikiSum dataset provides how-to articles from wikihow.com and their summaries, written as a coherent paragraph.
 From paper: "WikiSum: Coherent Summarization Dataset for Efficient Human-Evaluation" by N. Cohen et al.
 See: https://aclanthology.org/2021.acl-short.28.pdf
 See: https://registry.opendata.aws/wikisum/
"""
_CITATION = """\
    @inproceedings{cohen-etal-2021-wikisum,
    title = "{W}iki{S}um: Coherent Summarization Dataset for Efficient Human-Evaluation",
    author = "Cohen, Nachshon  and
      Kalinsky, Oren  and
      Ziser, Yftah  and
      Moschitti, Alessandro",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.28",
    doi = "10.18653/v1/2021.acl-short.28",
    pages = "212--219",
    abstract = "Recent works made significant advances on summarization tasks, facilitated by summarization datasets. Several existing datasets have the form of coherent-paragraph summaries. However, these datasets were curated from academic documents that were written for experts, thus making the essential step of assessing the summarization output through human-evaluation very demanding. To overcome these limitations, we present a dataset based on article summaries appearing on the WikiHow website, composed of how-to articles and coherent-paragraph summaries written in plain language. We compare our dataset attributes to existing ones, including readability and world-knowledge, showing our dataset makes human evaluation significantly easier and thus, more effective. A human evaluation conducted on PubMed and the proposed dataset reinforces our findings.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"

class WikiSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiSum."""

    def __init__(self, **kwargs):
        """BuilderConfig for WikiSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiSumConfig, self).__init__(**kwargs)


class WikiSumDataset(datalabs.GeneratorBasedBuilder):
    """WikiSum Dataset."""
    _FILE_URL = "https://wikisum.s3.amazonaws.com/WikiSumDataset.zip"
    BUILDER_CONFIGS = [
        WikiSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="WikiSum dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

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
            homepage=None,
            citation=_CITATION,
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._FILE_URL)
        f_path = os.path.join(f_path, "./WikiSumDataset/WikiSumDataset.jsonl")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": f_path, "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": f_path, "split": "dev"}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": f_path, "split": "test"}
            ),
        ]

    def _generate_examples(self, f_path, split):
        """Generate WikiSum examples."""
        cnt = 0
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["fold"] == split:
                    cnt += 1
                    yield cnt, {
                        _ARTICLE: data["article"],
                        _ABSTRACT: data["summary"],
                    }