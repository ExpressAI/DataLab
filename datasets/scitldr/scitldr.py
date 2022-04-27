"""SciTldr: Extreme Summarization of Scientific Documents."""
import json
import datalabs
from datalabs.tasks import Summarization

_CITATION = """\
@inproceedings{cachola-etal-2020-tldr,
    title = "{TLDR}: Extreme Summarization of Scientific Documents",
    author = "Cachola, Isabel  and
      Lo, Kyle  and
      Cohan, Arman  and
      Weld, Daniel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.428",
    doi = "10.18653/v1/2020.findings-emnlp.428",
    pages = "4766--4777",
    abstract = "We introduce TLDR generation, a new form of extreme summarization, for scientific papers. TLDR generation involves high source compression and requires expert background knowledge and understanding of complex domain-specific language. To facilitate study on this task, we introduce SCITLDR, a new multi-target dataset of 5.4K TLDRs over 3.2K papers. SCITLDR contains both author-written and expert-derived TLDRs, where the latter are collected using a novel annotation protocol that produces high-quality summaries while minimizing annotation burden. We propose CATTS, a simple yet effective learning strategy for generating TLDRs that exploits titles as an auxiliary training signal. CATTS improves upon strong baselines under both automated metrics and human evaluations. Data and code are publicly available at https://github.com/allenai/scitldr.",
}
"""

_DESCRIPTION = """\
We release SCITLDR, a new multi-target dataset of 5,411 TLDRs over 3,229 scientific papers. 
SCITLDR contains both author-written and expertderived TLDRs, 
where the latter are collected using a novel annotation protocol that 
produces highquality summaries while avoiding the burden of reading the full paper. 
TLDR generation seeks to produce an extreme (single sentence) summary (Narayan et al., 2018) 
given the entire paper.
see: https://aclanthology.org/2020.findings-emnlp.428.pdf
"""

_HOMEPAGE = "https://github.com/allenai/scitldr"
_LICENSE = "Apache License 2.0"
_ABSTRACT = "summary"
_ARTICLE = "text"


class SciTldrConfig(datalabs.BuilderConfig):
    """BuilderConfig for SciTldr."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for SciTldr.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SciTldrConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class SciTldrDataset(datalabs.GeneratorBasedBuilder):
    """SciTldr Dataset."""
    _FILE_ID = {
        "train": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-FullText/train.jsonl",
        "dev": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-FullText/dev.jsonl",
        "test": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-FullText/test.jsonl"}

    BUILDER_CONFIGS = [
        SciTldrConfig(
            name="tldr-auth",
            version=datalabs.Version("1.0.0"),
            description="Scientific document summarization dataset. TLDR-auth: TLDRs written from the perspective of the authors.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        SciTldrConfig(
            name="tldr-pr",
            version=datalabs.Version("1.0.0"),
            description="Scientific document summarization dataset. TLDR-pr: TLDRs written from the perspective of the peer reviewers.",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "tldr-auth"

    def _info(self):
        if self.config.name == "tldr-auth":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        elif self.config.name == "tldr-pr":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=self.VERSION,
            languages=["en"],
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        train_f_path = dl_manager.download(self._FILE_ID["train"])
        dev_f_path = dl_manager.download(self._FILE_ID["dev"])
        test_f_path = dl_manager.download(self._FILE_ID["test"])

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": dev_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate SciTldr examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        datas = []
        for line in lines:
            data = json.loads(line)
            article = " ".join(data["source"]).strip()
            if self.config.name == "tldr-auth":
                summary = data["target"][0].strip()  # datalabs.Value("string")
            elif self.config.name == "tldr-pr":
                summary = data["target"][1:]  # datalabs.Sequence(datalabs.Value("string"))

            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: article,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
