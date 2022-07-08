"""Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization"""
import os
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{deutsch-roth-2019-summary,
    title = "Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization",
    author = "Deutsch, Daniel  and
      Roth, Dan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1386",
    doi = "10.18653/v1/D19-1386",
    pages = "3720--3729"
}
"""

_DESCRIPTION = """\
We collected a new large-scale summary cloze dataset from Wikipedia, called the WIKICITE dataset. 
Each paragraph in Wikipedia can be viewed as a topic-focused summary of the references cited within 
the paragraph, where the topic is defined as the article title and section
headings. The citations provide supervision at the sentence-level that indicates where the content of
the preceding sentence came from. We scraped hundreds of thousands of Wikipedia articles and
corresponding references to collect nearly 500k summary cloze instances.
see: https://aclanthology.org/D19-1386.pdf
"""

_HOMEPAGE = "https://github.com/danieldeutsch/wikicite"
_ABSTRACT = "summary"
_ARTICLE = "texts"
_KEY = "query"


class WikiCiteConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiCite."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for WikiCite.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiCiteConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class WikiCiteDataset(datalabs.GeneratorBasedBuilder):
    """WikiCite Dataset."""

    BUILDER_CONFIGS = [
        WikiCiteConfig(
            name="query-multi-doc",
            version=datalabs.Version("1.1.0"),
            description="Large-scale summary cloze dataset from Wikipedia.",
            task_templates=[get_task(TaskType.query_multi_doc_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT,
                guidance_column=_KEY)]
        )
    ]
    DEFAULT_CONFIG_NAME = "query-multi-doc"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                    _KEY: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        train_url = "https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/train.tokenized.v1.1.jsonl.gz"
        valid_url = "https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/valid.tokenized.v1.1.jsonl.gz"
        test_url = "https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/test.tokenized.v1.1.jsonl.gz"

        train_f_path = dl_manager.download_and_extract(train_url)
        valid_f_path = dl_manager.download_and_extract(valid_url)
        test_f_path = dl_manager.download_and_extract(test_url)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": train_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={"f_path": valid_f_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate WikiCite examples."""
        f = open(f_path, encoding="utf-8")
        lines = f.readlines()
        original_datas = []
        for line in lines:
            line = line.strip()
            data = json.loads(line)
            original_datas.append(data)

        datas = []
        for original_data in original_datas:
            # query
            page_title = original_data["page_title"].strip()  # str
            headings = original_data["headings"]  # list
            query = page_title + " ".join(headings)

            # summary
            summary = original_data["cloze"]  # cloze summary

            # documents
            texts = []
            left_context = " ".join(original_data["left_context"]).strip()
            texts.append(left_context)
            documents = original_data["documents"]  # cited documents
            for document in documents:
                paragraphs = document["paragraphs"]
                sentences = [sentence.strip() for paragraph in paragraphs for sentence in paragraph]
                text = " ".join(sentences)
                texts.append(text)

            datas.append({"query": query, "texts": texts, "summary": summary})

        for id_, data in enumerate(datas):
            raw_feature_info = {
                _KEY: data["query"],
                _ARTICLE: data["texts"],
                _ABSTRACT: data["summary"]

            }
            yield id_, raw_feature_info
