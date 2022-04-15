"""SSN (Semantic Scholar Network): Multi-Document Scientific Papers Summarization."""
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
@article{An_Zhong_Chen_Wang_Qiu_Huang_2021, 
    title = {Enhancing Scientific Papers Summarization with Citation Graph}, 
    volume = {35}, 
    url = {https://ojs.aaai.org/index.php/AAAI/article/view/17482}, 
    abstractNote = {Previous work for text summarization in scientific domain mainly focused on the content of the input document, but seldom considering its citation network. 
    However, scientific papers are full of uncommon domain-specific terms, making it almost impossible for the model to understand its true meaning without the help of the relevant research community. 
    In this paper, we redefine the task of scientific papers summarization by utilizing their citation graph and propose a citation graph-based summarization model CGSum which can incorporate the information of both the source paper and its references. 
    In addition, we construct a novel scientific papers summarization dataset Semantic Scholar Network (SSN) which contains 141K research papers in different domains and 661K citation relationships. The entire dataset constitutes a large connected citation graph. 
    Extensive experiments show that our model can achieve competitive performance when compared with the pretrained models even with a simple architecture. 
    The results also indicates the citation graph is crucial to better understand the content of papers and generate high-quality summaries.}, 
    number={14}, 
    journal = {Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author = {An, Chenxin and Zhong, Ming and Chen, Yiran and Wang, Danqing and Qiu, Xipeng and Huang, Xuanjing}, 
    year = {2021}, 
    month = {May}, 
    pages = {12498-12506} 
}
"""

_DESCRIPTION = """\
we construct a novel scientific papers summarization dataset Semantic Scholar Network (SSN)
which contains 141K research papers in different domains and 661K citation relationships. 
The entire dataset constitutes a large connected citation graph.
We divide the enhanced summarization task into 2 settings: 
(1) transductive: during training, models can access to all the nodes and edges in the
whole dataset including papers (excluding abstracts) in the test set. 
(2) inductive: papers in the test set are from a totally new graph
which means all test nodes cannot be used during training.
see: https://arxiv.org/abs/2104.03057
"""

_HOMEPAGE = "https://github.com/ChenxinAn-fdu/CGSum"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class SSNConfig(datalabs.BuilderConfig):
    """BuilderConfig for SSN."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for SSN.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SSNConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class SSNDataset(datalabs.GeneratorBasedBuilder):
    """SSN Dataset."""

    _FILE_ID = {
        "transductive": _gdrive_url("1SdrWHoDRU0-P21b4LM42SwFt8zx3d4F2"),
        "inductive": _gdrive_url("1GJOkm3iQf7kBxme1ZFuwYPeTV3J8QV17"),
        "papers": _gdrive_url("1P5viA8hMm19n-Ia3k9wZyQTEloCk2gMJ")
    }

    BUILDER_CONFIGS = [
        SSNConfig(
            name="transductive-document",
            version=datalabs.Version("1.0.0"),
            description="SSN dataset for scientific multi-document summarization, single document version. Transductive: during training, models can access to all the nodes and edges in the whole dataset including papers (excluding abstracts) in the test set.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        SSNConfig(
            name="transductive-multidoc",
            version=datalabs.Version("1.0.0"),
            description="SSN dataset for scientific multi-document summarization, multi-document version. Inductive: papers in the test set are from a totally new graph which means all test nodes cannot be used during training.",
            task_templates=[MultiDocSummarization(
                text_column=_MDS_TEXT_COLUMN, summary_column=_ABSTRACT)]
        ),
        SSNConfig(
            name="inductive-document",
            version=datalabs.Version("1.0.0"),
            description="SSN dataset for scientific multi-document summarization, single document version. Transductive: during training, models can access to all the nodes and edges in the whole dataset including papers (excluding abstracts) in the test set.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        SSNConfig(
            name="inductive-multidoc",
            version=datalabs.Version("1.0.0"),
            description="SSN dataset for scientific multi-document summarization, multi-document version. Inductive: papers in the test set are from a totally new graph which means all test nodes cannot be used during training.",
            task_templates=[MultiDocSummarization(
                text_column=_MDS_TEXT_COLUMN, summary_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "transductive-document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_dataset = {}

        if "document" in self.config.name:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                        get_schema_of_sample_level_features)
        elif "multidoc" in self.config.name:
            features_sample = datalabs.Features(
                {
                    _MDS_TEXT_COLUMN: {
                        "introduction": datalabs.Value("string"),
                        "references": datalabs.Sequence(datalabs.Value("string"))
                    },
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

        papers_f_path = dl_manager.download_and_extract(self._FILE_ID["papers"])
        papers_f_path = os.path.join(papers_f_path, "SSN/papers.SSN.jsonl")

        if "inductive" in self.config.name:
            f_path = dl_manager.download_and_extract(self._FILE_ID["inductive"])
        elif "transductive" in self.config.name:
            f_path = dl_manager.download_and_extract(self._FILE_ID["transductive"])

        type = self.config.name.split("-")[0].strip()
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "{}/train.jsonl".format(type)),
                            "papers_f_path": papers_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": os.path.join(f_path, "{}/val.jsonl".format(type)),
                            "papers_f_path": papers_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": os.path.join(f_path, "{}/test.jsonl".format(type)),
                            "papers_f_path": papers_f_path}
            ),
        ]

    def _generate_examples(self, f_path, papers_f_path):
        """Generate SSN examples."""

        # read whole papers information
        id2abstract = {}
        with open(papers_f_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                paper = json.loads(line)
                id2abstract[paper["paper_id"]] = " ".join(paper["abstract"])

        datas = []
        with open(f_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                paper = json.loads(line)
                references = paper["references"]
                ref_abstracts = []
                for reference_id in references:
                    ref_abstract = id2abstract[reference_id].strip()
                    ref_abstracts.append(ref_abstract)
                introduction = paper["introduction"].strip()
                summary = paper["abstract"].strip()
                datas.append({"ref_abstracts": ref_abstracts, "introduction": introduction, "summary": summary})

        if "document" in self.config.name:
            for id_, data in enumerate(datas):
                raw_feature_info = {
                    _ARTICLE: data["introduction"] + " " + " ".join(data["ref_abstracts"]).strip(),
                    _ABSTRACT: data["summary"]
                }
                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
        elif "multidoc" in self.config.name:
            for id_, data in enumerate(datas):
                raw_feature_info = {
                    _MDS_TEXT_COLUMN: {
                        "introduction": data["introduction"],
                        "references": data["ref_abstracts"]
                    },
                    _ABSTRACT: data["summary"]
                }
                yield id_, raw_feature_info
