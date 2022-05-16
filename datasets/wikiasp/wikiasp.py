"""WikiAsp:  A Dataset for Multi-domain Aspect-based Summarization"""
import os
import datalabs
from datalabs import get_task, TaskType
import json

_CITATION = """\
@article{10.1162/tacl_a_00362,
    author = {Hayashi, Hiroaki and Budania, Prashant and Wang, Peng and Ackerson, Chris and Neervannan, Raj and Neubig, Graham},
    title = "{WikiAsp: A Dataset for Multi-domain Aspect-based Summarization}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {211-225},
    year = {2021},
    month = {03},
    abstract = "{Aspect-based summarization is the task of generating focused summaries based on specific points of interest. Such summaries aid efficient analysis of text, such as quickly understanding reviews or opinions from different angles. However, due to large differences in the type of aspects for different domains (e.g., sentiment, product features), the development of previous models has tended to be domain-specific. In this paper, we propose WikiAsp,1 a large-scale dataset for multi-domain aspect- based summarization that attempts to spur research in the direction of open-domain aspect-based summarization. Specifically, we build the dataset using Wikipedia articles from 20 different domains, using the section titles and boundaries of each article as a proxy for aspect annotation. We propose several straightforward baseline models for this task and conduct experiments on the dataset. Results highlight key challenges that existing summarization models face in this setting, such as proper pronoun handling of quoted sources and consistent explanation of time-sensitive events.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00362},
    url = {https://doi.org/10.1162/tacl\_a\_00362},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00362/1924027/tacl\_a\_00362.pdf},
}
"""

_DESCRIPTION = """\
WikiAsp is a multi-domain, aspect-based summarization dataset in the encyclopedic domain. In this task, models are asked to summarize cited reference documents of a Wikipedia article into aspect-based summaries. Each of the 20 domains include 10 domain-specific pre-defined aspects.
see: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00362/98088/WikiAsp-A-Dataset-for-Multi-domain-Aspect-based
"""

_HOMEPAGE = "https://github.com/neulab/wikiasp"
_ABSTRACT = "summary"
_ARTICLE = "text"
_KEY = "query"


class WikiAspConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiAsp."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WikiAsp.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiAspConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WikiAspDataset(datalabs.GeneratorBasedBuilder):
    """WikiAsp Dataset."""
    _DOMAINS = [
        "Album", 
        "Animal",
        "Artist",
        "Building",
        "Company",
        "EducationalInstitution",
        "Event",
        "Film",
        "Group",
        "HistoricPlace",
        "Infrastructure",
        "MeanOfTransportation",
        "OfficeHolder",
        "Plant",
        "Single",
        "SoccerPlayer",
        "Software",
        "TelevisionShow",
        "Town",
        "WrittenWork"
        ]
    BUILDER_CONFIGS = list([WikiAspConfig(
            name=x,
            version=datalabs.Version("1.0.0"),
            description=f"WikiAsp Dataset for aspect-based summarization, {x} split",
            task_templates=[get_task(TaskType.query_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column=_KEY)]
        ) for x in _DOMAINS])
    DEFAULT_CONFIG_NAME = "Album"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    _KEY: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=[self.config.name],
            task_templates=[get_task(TaskType.query_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column=_KEY),
            ],
        )

    def _split_generators(self, dl_manager):
        name = self.config.name
        url = f"http://phontron.com/download/wikiasp/{name}.tar.bz2"
        path = dl_manager.download_and_extract(url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": os.path.join(path, f"{name}/train.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": os.path.join(path, f"{name}/valid.jsonl")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": os.path.join(path, f"{name}/test.jsonl")},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate WikiAsp examples."""
        cnt = 0
        with open(f_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = " ".join(data["inputs"])
                for t in data["targets"]:
                    yield cnt, {_ARTICLE: text, _ABSTRACT: t[1], _KEY: t[0]}
                    cnt += 1
                