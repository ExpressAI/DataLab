import json
import os
import datalabs
from datalabs.tasks import Summarization, MultiDocSummarization
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN
import gzip

_DESCRIPTION = """
 Multi-XScience, a large-scale multi-document summarization dataset created from scientific articles. Multi-XScience introduces a challenging multi-document summarization task: writing therelated-work section of a paper based on itsabstract and the articles it references.
 From paper: "Multi-XScience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles" by Y. Lu et al.
 See: https://aclanthology.org/2020.emnlp-main.648/
 See: https://github.com/yaolu/Multi-XScience
"""
_CITATION = """\
    @inproceedings{lu-etal-2020-multi-xscience,
    title = "Multi-{XS}cience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles",
    author = "Lu, Yao  and
      Dong, Yue  and
      Charlin, Laurent",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.648",
    doi = "10.18653/v1/2020.emnlp-main.648",
    pages = "8068--8074",
    abstract = "Multi-document summarization is a challenging task for which there exists little large-scale datasets. We propose Multi-XScience, a large-scale multi-document summarization dataset created from scientific articles. Multi-XScience introduces a challenging multi-document summarization task: writing the related-work section of a paper based on its abstract and the articles it references. Our work is inspired by extreme summarization, a dataset construction protocol that favours abstractive modeling approaches. Descriptive statistics and empirical results{---}using several state-of-the-art models trained on the Multi-XScience dataset{---}reveal that Multi-XScience is well suited for abstractive models.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"
_MDS_SEP = "|||||"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"

class MultiXScienceConfig(datalabs.BuilderConfig):
    """BuilderConfig for MultiXScience."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MultiXScience.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiXScienceConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MultiXScienceDataset(datalabs.GeneratorBasedBuilder):
    """MultiXScience Dataset."""
    _TRAIN_URL = "https://raw.githubusercontent.com/yaolu/Multi-XScience/master/data/train.json.gz"
    _VAL_URL = "https://raw.githubusercontent.com/yaolu/Multi-XScience/master/data/val.json.gz"
    _TEST_URL = "https://raw.githubusercontent.com/yaolu/Multi-XScience/master/data/test.json.gz"
    BUILDER_CONFIGS = [
        MultiXScienceConfig(
            name="single-document",
            version=datalabs.Version("1.0.0"),
            description="MultiXScience dataset for summarization, single document summarization version",
            task_templates=[Summarization(text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        MultiXScienceConfig(
            name="multi-document",
            version=datalabs.Version("1.0.0"),
            description="MultiXScience dataset for summarization, multi-document summarization version",
            task_templates=[MultiDocSummarization(text_column=_MDS_TEXT_COLUMN, summary_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "multi-document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        if "single" in self.config.name:
            features = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        else:
            features = datalabs.Features(
                {
                    _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://github.com/yaolu/Multi-XScience",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._TRAIN_URL)
        val_path = dl_manager.download(self._VAL_URL)
        test_path = dl_manager.download(self._TEST_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate MultiXScience examples."""
        with gzip.open(f_path) as f:
            data = json.load(f)
        for (id_, x) in enumerate(data):
            summary = x["related_work"]
            text = [x["abstract"]]
            for k in x["ref_abstract"]:
                if len(x["ref_abstract"][k]["abstract"]) > 0:
                    # skip empty abstract
                    text.append("{}: {}".format(k, x["ref_abstract"][k]["abstract"]))
            if "single" in self.config.name:
                text = _MDS_SEP.join(text)
                yield id_, {
                    _ARTICLE: text,
                    _ABSTRACT: summary,
                }
            else:
                yield id_, {
                    _MDS_TEXT_COLUMN: text,
                    _ABSTRACT: summary,
                }