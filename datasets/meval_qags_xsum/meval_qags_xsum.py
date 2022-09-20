import json

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Sequence, Value

_DESCRIPTION = """\
QAGS20 collected 235 test outputs on the CNNDM dataset and 239 test outputs on the XSUM dataset.
"""

_CITATION = """\
@inproceedings{wang-etal-2020-asking,
    title = "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries",
    author = "Wang, Alex  and
      Cho, Kyunghyun  and
      Lewis, Mike",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.450",
    doi = "10.18653/v1/2020.acl-main.450",
    pages = "5008--5020",
    abstract = "Practical applications of abstractive summarization models are limited by frequent factual inconsistencies with respect to their input. Existing automatic evaluation metrics for summarization are largely insensitive to such errors. We propose QAGS (pronounced {``}kags{''}), an automatic evaluation protocol that is designed to identify factual inconsistencies in a generated summary. QAGS is based on the intuition that if we ask questions about a summary and its source, we will receive similar answers if the summary is factually consistent with the source. To evaluate QAGS, we collect human judgments of factual consistency on model-generated summaries for the CNN/DailyMail (Hermann et al., 2015) and XSUM (Narayan et al., 2018) summarization datasets. QAGS has substantially higher correlations with these judgments than other automatic evaluation metrics. Also, QAGS offers a natural form of interpretability: The answers and questions generated while computing QAGS indicate which tokens of a summary are inconsistent and why. We believe QAGS is a promising tool in automatically generating usable and factually consistent text. Code for QAGS will be available at https://github.com/W4ngatang/qags.",
}
"""

_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/meval/qags_xsum/data.jsonl"


class MevalQAGSXSUM(datalabs.GeneratorBasedBuilder):
    def _info(self):
        features = datalabs.Features(
            {
                "source": Value("string"),
                "references": Sequence(Value("string")),
                "hypotheses": Sequence({
                    "system_name": Value("string"),
                    "hypothesis": Value("string"),
                }
                ),
                "scores": Sequence({
                    "factuality": Value("float64"),
                })
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage="https://aclanthology.org/2020.acl-main.450/",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.nlg_meta_evaluation)(
                    source_column="source",
                    hypotheses_column="hypothesis",
                    references_column="references",
                    scores_column="scores",
                )
            ]
        )

    def _split_generators(self, dl_manager):
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """ Generate QAGS-XSUM examples."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = line.strip()
                line = json.loads(line)
                source, hypotheses_scores, references = line["source"], line["hypotheses"], line["references"]

                hypotheses = [ {"system_name":x["system_name"],
                                "hypothesis":x["hypothesis"]} for x in hypotheses_scores]
                scores = [x["scores"] for x in hypotheses_scores]
                yield id_, {
                    "source": source,
                    "hypotheses": hypotheses,
                    "references": references,
                    "scores": scores,
                }
