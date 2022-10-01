import json

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Sequence, Value

_DESCRIPTION = """\
Rank19 is used to meta-evaluate factuality metrics. It is a collection of 373 triples of a 
source sentence with two summary sentences, one correct and one incorrect.
"""

_CITATION = """\
@inproceedings{falke-etal-2019-ranking,
    title = "Ranking Generated Summaries by Correctness: An Interesting but Challenging Application for Natural Language Inference",
    author = "Falke, Tobias  and
      Ribeiro, Leonardo F. R.  and
      Utama, Prasetya Ajie  and
      Dagan, Ido  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1213",
    doi = "10.18653/v1/P19-1213",
    pages = "2214--2220",
    abstract = "While recent progress on abstractive summarization has led to remarkably fluent summaries, factual errors in generated summaries still severely limit their use in practice. In this paper, we evaluate summaries produced by state-of-the-art models via crowdsourcing and show that such errors occur frequently, in particular with more abstractive models. We study whether textual entailment predictions can be used to detect such errors and if they can be reduced by reranking alternative predicted summaries. That leads to an interesting downstream application for entailment models. In our experiments, we find that out-of-the-box entailment models trained on NLI datasets do not yet offer the desired performance for the downstream task and we therefore release our annotations as additional test data for future extrinsic evaluations of NLI.",
}
"""

_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/meval/rank19/data.jsonl"


class MevalRank19Config(datalabs.BuilderConfig):


    def __init__(
        self,
        evaluation_aspect = None,
        **kwargs
    ):
        super(MevalRank19Config, self).__init__(**kwargs)
        self.evaluation_aspect = evaluation_aspect



class MevalRank19(datalabs.GeneratorBasedBuilder):

    evaluation_aspects = [
        "factuality",
    ]
    BUILDER_CONFIGS = [MevalRank19Config(
        name=aspect,
        version=datalabs.Version("1.0.0"),
        evaluation_aspect=aspect
    ) for aspect in evaluation_aspects]


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
                "scores": Sequence(Value("float")),
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage="https://aclanthology.org/P19-1213/",
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
        """ Generate Rank19 examples."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = line.strip()
                line = json.loads(line)
                source, hypotheses_scores, references = line["source"], line["hypotheses"], line["references"]

                hypotheses = [ {"system_name":x["system_name"],
                                "hypothesis":x["hypothesis"]} for x in hypotheses_scores]
                scores = [x["scores"][self.config.name] for x in hypotheses_scores]
                yield id_, {
                    "source": source,
                    "hypotheses": hypotheses,
                    "references": references,
                    "scores": scores,
                }
