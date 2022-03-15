import json
import os
import datalabs
from datalabs.tasks import Summarization

_DESCRIPTION = """
 A peer review dataset collected from ICLR (2017~2020) and NeurIPS (2016-2019).
 From paper: Can We Automate Scientific Reviewing?
 See: https://arxiv.org/pdf/2102.00176.pdf
"""

_CITATION = """\
@misc{yuan2021automate,
      title={Can We Automate Scientific Reviewing?}, 
      author={Weizhe Yuan and Pengfei Liu and Graham Neubig},
      year={2021},
      eprint={2102.00176},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_TRAIN_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/asap_review/new_train.jsonl"
_VALIDATION_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/asap_review/new_validation.jsonl"
_TEST_DOWNLOAD_URL = "https://datalab-hub.s3.amazonaws.com/asap_review/new_test.jsonl"


class ASAPReviewDataset(datalabs.GeneratorBasedBuilder):
    """ ASAP-Review Dataset. """

    def _info(self):
        features_dataset = {}
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "paper_id": datalabs.Value("string"),
                    "title": datalabs.Value("string"),
                    "abstract": datalabs.Value("string"),
                    "content": datalabs.Sequence({
                        "heading": datalabs.Value("string"),
                        "text": datalabs.Value("string")
                    }),
                    "aspects": datalabs.Sequence({
                        "start_idx": datalabs.Value("int32"),
                        "end_idx": datalabs.Value("int32"),
                        "aspect": datalabs.Value("string")
                    }),
                    "review": datalabs.Value("string"),
                    "text": datalabs.Value("string")
                }
            ),
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage="https://github.com/neulab/ReviewAdvisor",
            citation=_CITATION,
            task_templates=[Summarization(text_column="text", summary_column="review")]
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        print(f"train_path: \t{train_path}")
        val_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        print(f"validation_path: \t{val_path}")
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        print(f"test_path: \t{test_path}")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": val_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate ArxivSummarization examples."""
        with open(filepath, errors="surrogateescape") as f:
            for id_, line in enumerate(f.readlines()):
                line = line.strip()
                data = json.loads(line)
                paper_id = data["id"]
                title = data["title"]
                abstract = data["abstract"]
                content = data["content"]
                aspects = data["aspects"]
                review = data["review"]
                text = data["text"]
                yield id_, {"paper_id": paper_id, "title": title, "abstract": abstract, "content": content,
                            "aspects": aspects, "review": review, "text": text}
