"""AnswerSumm: A Manually-Curated Dataset and Pipeline for Answer Summarization"""
import os
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{fabbri2021answersumm,
  title={AnswerSumm: A Manually-Curated Dataset and Pipeline for Answer Summarization},
  author={Fabbri, Alexander R and Wu, Xiaojian and Iyer, Srini and Li, Haoran and Diab, Mona},
  journal={arXiv preprint arXiv:2111.06474},
  year={2021}
}
"""

_DESCRIPTION = """\
This work introduces a novel dataset of 4,631 CQA threads for answer summarization curated by professional linguists.
see: https://arxiv.org/pdf/2111.06474.pdf
"""

_HOMEPAGE = "https://github.com/Alex-Fabbri/AnswerSumm"
_ABSTRACT = "summary"
_LICENSE = "cc-by-sa 4.0"
_ARTICLE = "texts"
_KEY = "query"


class AnswerSummConfig(datalabs.BuilderConfig):
    """BuilderConfig for AnswerSumm."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for AnswerSumm.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AnswerSummConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class AnswerSummDataset(datalabs.GeneratorBasedBuilder):
    """AnswerSumm Dataset."""

    BUILDER_CONFIGS = [
        AnswerSummConfig(
            name="query-multi-doc",
            version=datalabs.Version("1.0.0"),
            description="Dataset for answer summarization.",
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
            license=_LICENSE,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        train_url = "https://huggingface.co/datasets/alexfabbri/answersumm/resolve/main/train.jsonl"
        valid_url = "https://huggingface.co/datasets/alexfabbri/answersumm/resolve/main/validation.jsonl"
        test_url = "https://huggingface.co/datasets/alexfabbri/answersumm/resolve/main/test.jsonl"

        train_f_path = dl_manager.download(train_url)
        valid_f_path = dl_manager.download(valid_url)
        test_f_path = dl_manager.download(test_url)

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
        """Generate AnswerSumm examples."""
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
            query = original_data["question"]["question"].strip()

            # summary
            summary = " ".join(original_data["summaries"][0])

            # documents
            texts = []
            answers = original_data["answers"]
            for answer in answers:
                answer_sents = []
                for sent in answer["sents"]:
                    answer_sents.append(sent["text"])
                text = " ".join(answer_sents)
                texts.append(text)

            datas.append({"query": query, "texts": texts, "summary": summary})

        for id_, data in enumerate(datas):
            raw_feature_info = {
                _KEY: data["query"],
                _ARTICLE: data["texts"],
                _ABSTRACT: data["summary"]

            }
            yield id_, raw_feature_info
