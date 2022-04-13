""" GameWikiSum: a Novel Large Multi-Document Summarization Dataset """
import os
import json
import datalabs
from datalabs.tasks import Summarization, MultiDocSummarization
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
)
from datalabs.utils.more_features import (
    get_feature_schemas,
)

_CITATION = """\
@InProceedings{antognini-faltings:2020:LREC2,
  author    = {Antognini, Diego  and  Faltings, Boi},
  title     = {GameWikiSum: a Novel Large Multi-Document Summarization Dataset},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {6645--6650},
  abstract  = {Today's research progress in the field of multi-document summarization is obstructed by the small number of available datasets. Since the acquisition of reference summaries is costly, existing datasets contain only hundreds of samples at most, resulting in heavy reliance on hand-crafted features or necessitating additional, manually annotated data. The lack of large corpora therefore hinders the development of sophisticated models. Additionally, most publicly available multi-document summarization corpora are in the news domain, and no analogous dataset exists in the video game domain. In this paper, we propose GameWikiSum, a new domain-specific dataset for multi-document summarization, which is one hundred times larger than commonly used datasets, and in another domain than news. Input documents consist of long professional video game reviews as well as references of their gameplay sections in Wikipedia pages. We analyze the proposed dataset and show that both abstractive and extractive models can be trained on it. We release GameWikiSum for further research: https://github.com/Diego999/GameWikiSum.},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.820}
}
"""

_DESCRIPTION = """\
GameWikiSum, a new domain-specific (video game) dataset for multi-document summarization, 
which is one hundred times larger than commonly used datasets, and in another domain than news. 
Input documents consist of long professional video game reviews as well as references 
of their gameplay sections in Wikipedia pages.
see: https://aclanthology.org/2020.lrec-1.820/
"""

_HOMEPAGE = "https://github.com/Diego999/GameWikiSum"
_ABSTRACT = "summary"
_ARTICLE = "text"


class GameWikiSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for GameWikiSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for GameWikiSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GameWikiSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class GameWikiSumDataset(datalabs.GeneratorBasedBuilder):
    """GameWikiSum Dataset."""
    _FILE_ID = "http://lia.epfl.ch/Datasets/Full_GameWiki.zip"

    BUILDER_CONFIGS = [
        GameWikiSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="GameWikiSum dataset for multi-document summarization, single document version.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        GameWikiSumConfig(
            name="multidoc",
            version=datalabs.Version("1.0.0"),
            description="GameWikiSum dataset for multi-document summarization, multi-document version.",
            task_templates=[MultiDocSummarization(
                text_column=_MDS_TEXT_COLUMN, summary_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_dataset = {}

        if self.config.name == "document":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                        get_schema_of_sample_level_features)
        elif self.config.name == "multidoc":
            features_sample = datalabs.Features(
                {
                    _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(self._FILE_ID)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "train.json")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": os.path.join(f_path, "val.json")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": os.path.join(f_path, "test.json")}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate GameWikiSum examples."""

        with open(f_path, encoding="utf-8") as f:
            data_segs = json.load(f)

        all_datas = []
        for datas in data_segs:
            for data in datas:
                preprocessed_paragraphs_per_reviews = data["preprocessed_paragraphs_per_reviews"]  # list of list
                preprocessed_summary = data["preprocessed_summary"]  # string
                all_datas.append((preprocessed_paragraphs_per_reviews, preprocessed_summary))

        for (id_, data) in enumerate(all_datas):

            reviews = data[0]
            summary = data[1]

            if self.config.name == "document":

                article = " ".join([" ".join(sentences) for sentences in reviews])

                raw_feature_info = {
                    _ARTICLE: article,
                    _ABSTRACT: summary
                }

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
            elif self.config.name == "multidoc":
                article = [" ".join(sentences) for sentences in reviews]
                yield id_, {
                    _MDS_TEXT_COLUMN: article,
                    _ABSTRACT: summary,
                }
