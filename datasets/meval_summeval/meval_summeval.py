import json

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Sequence, Value

_DESCRIPTION = """\
SummEval is a collection of human judgments of model-generated summaries on the CNNDM dataset 
annotated by both expert judges and crowd-source workers. Each system generated summary is 
gauged through the lens of coherence, consistency, fluency and relevance.
"""

_CITATION = """\
@article{fabbri2020summeval,
  title={SummEval: Re-evaluating Summarization Evaluation},
  author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal={arXiv preprint arXiv:2007.12626},
  year={2020}
}
"""

_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/meval/summeval/test.jsonl"


class MevalSummEvalConfig(datalabs.BuilderConfig):


    def __init__(
        self,
        evaluation_aspect = None,
        **kwargs
    ):
        super(MevalSummEvalConfig, self).__init__(**kwargs)
        self.evaluation_aspect = evaluation_aspect


class MevalSummEval(datalabs.GeneratorBasedBuilder):

    evaluation_aspects = [
        "coherence",
        "fluency",
        "consistency",
        "relevance"
    ]

    BUILDER_CONFIGS = [MevalSummEvalConfig(
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
                    "hypothesis": Value("string")
                }
                ),
                "scores": Sequence(Value("float")),
            }
        )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage="https://github.com/Yale-LILY/SummEval",
            citation=_CITATION,
            languages=["en"],
            task_templates=[
                get_task(TaskType.meta_evaluation_nlg)(
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
        """ Generate SummEval examples."""
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
