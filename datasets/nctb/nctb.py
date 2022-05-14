"""NCTB Bengali abstractive summarization dataset."""
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{chowdhury-etal-2021-unsupervised,
    title = "Unsupervised Abstractive Summarization of {B}engali Text Documents",
    author = "Chowdhury, Radia Rayan  and
      Nayeem, Mir Tafseer  and
      Mim, Tahsin Tasnim  and
      Chowdhury, Md. Saifur Rahman  and
      Jannat, Taufiqul",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.224",
    doi = "10.18653/v1/2021.eacl-main.224",
    pages = "2612--2619",
    abstract = "Abstractive summarization systems generally rely on large collections of document-summary pairs. However, the performance of abstractive systems remains a challenge due to the unavailability of the parallel data for low-resource languages like Bengali. To overcome this problem, we propose a graph-based unsupervised abstractive summarization system in the single-document setting for Bengali text documents, which requires only a Part-Of-Speech (POS) tagger and a pre-trained language model trained on Bengali texts. We also provide a human-annotated dataset with document-summary pairs to evaluate our abstractive model and to support the comparison of future abstractive summarization systems of the Bengali Language. We conduct experiments on this dataset and compare our system with several well-established unsupervised extractive summarization systems. Our unsupervised abstractive summarization model outperforms the baselines without being exposed to any human-annotated reference summaries.",
}
"""

_DESCRIPTION = """\
We also introduce a highly abstractive dataset with document-summary pairs, 
which is written by professional summary writers of National Curriculum and Textbook Board (NCTB).
We collected the human written document-summary pairs from the several printed copy of NCTB books.
see: https://aclanthology.org/2021.eacl-main.224.pdf
"""

_HOMEPAGE = "https://github.com/tafseer-nayeem/BengaliSummarization"
_ARTICLE = "text"
_ABSTRACT = "summary"


def get_article_summary_urls():
    """
    The NCTB dataset maintained in the Github repo is listed as separate files.
    For more information, see: https://github.com/tafseer-nayeem/BengaliSummarization
    """
    article_urls = []
    summary_urls = []
    root_url = "https://raw.githubusercontent.com/tafseer-nayeem/BengaliSummarization/main/Dataset/NCTB/"
    for index in range(1, 140):
        article_url = os.path.join(root_url, "Source/{}.txt".format(str(index)))
        summary_url = os.path.join(root_url, "Summary/{}.txt".format(str(index)))
        article_urls.append(article_url)
        summary_urls.append(summary_url)
    return article_urls, summary_urls


class NCTBConfig(datalabs.BuilderConfig):
    """BuilderConfig for NCTB."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for NCTB.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NCTBConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class NCTBDataset(datalabs.GeneratorBasedBuilder):
    """NCTB Dataset."""
    BUILDER_CONFIGS = [
        NCTBConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="NCTB, An Abstractive Bengali Summarization Dataset.",
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
            languages=["bn"],  # Bengali: https://huggingface.co/languages#:~:text=51-,Bengali,-bn
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )

    def _split_generators(self, dl_manager):
        article_urls, summary_urls = get_article_summary_urls()
        f_article_paths = dl_manager.download(article_urls)
        f_summary_paths = dl_manager.download(summary_urls)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_article_paths": f_article_paths, "f_summary_paths": f_summary_paths}
            ),
        ]

    def _generate_examples(self, f_article_paths, f_summary_paths):
        """Generate NCTB examples."""
        datas = []
        for f_article_path, f_summary_path in zip(f_article_paths, f_summary_paths):
            article_f = open(f_article_path, encoding="utf-8")
            summary_f = open(f_summary_path, encoding="utf-8")
            article = article_f.readlines()[0].strip()
            summary = summary_f.readlines()[0].strip()
            datas.append((article, summary))

        for id_, (article, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: article,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
