import json

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Sequence, Value

_DESCRIPTION = """\
SFRES is a meta-evaluation datasets in the data-to-text domain. It provides 
information about restaurants in San Francisco. Each sample in it consists 
of one meaning representation, multiple references, and utterances generated 
by different systems.
"""

_CITATION = """\
@inproceedings{wen-etal-2015-semantically,
    title = "Semantically Conditioned {LSTM}-based Natural Language Generation for Spoken Dialogue Systems",
    author = "Wen, Tsung-Hsien  and
      Ga{\v{s}}i{\'c}, Milica  and
      Mrk{\v{s}}i{\'c}, Nikola  and
      Su, Pei-Hao  and
      Vandyke, David  and
      Young, Steve",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1199",
    doi = "10.18653/v1/D15-1199",
    pages = "1711--1721",
}
"""

_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/meval/sfres/test.jsonl"


class MevalSFRESConfig(datalabs.BuilderConfig):


    def __init__(
        self,
        evaluation_aspect = None,
        **kwargs
    ):
        super(MevalSFRESConfig, self).__init__(**kwargs)
        self.evaluation_aspect = evaluation_aspect


class MevalSFRES(datalabs.GeneratorBasedBuilder):

    evaluation_aspects = [
        "naturalness",
        "informativeness",
        "quality"
    ]

    BUILDER_CONFIGS = [MevalSFRESConfig(
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
            homepage="https://github.com/jeknov/EMNLP_17_submission",
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
        """ Generate SFRES examples."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = line.strip()
                line = json.loads(line)
                source, hypothesis, references, scores = line["source"], line["hypothesis"], line["references"], line[
                    "scores"]
                yield id_, {
                    "source": source,
                    "hypotheses": [{
                        "system_name": "Unknown",
                        "hypothesis": hypothesis
                    }],
                    "references": references,
                    "scores": [scores[self.config.name]],
                }
