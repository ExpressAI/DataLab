import json
import os
import datalabs
from datalabs.tasks import Summarization
from tqdm import tqdm

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level_asap,
    get_schema_of_sample_level_features_asap,
    )
from datalabs.utils.more_features import (
    get_feature_schemas,
)



_DESCRIPTION = """
 A peer review dataset with more fine-grained aspect annotation (summary, motivation, originality, soundness, 
 substance, replicability), also known as, Aspect-enhanced Review Dataset. It collected from ICLR (2017~2020) and NeurIPS (2016-2019).
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
        features_sample = datalabs.Features(
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
            )

        if self.feature_expanding:
            features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                    get_schema_of_sample_level_features_asap)

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            languages=["en"],
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
            for id_, line in tqdm(enumerate(f.readlines())):
                line = line.strip()
                data = json.loads(line)
                paper_id = data["id"]
                title = data["title"]
                abstract = data["abstract"]
                content = data["content"]
                aspects = data["aspects"]
                review = data["review"]
                text = data["text"]

                raw_feature_info = {"paper_id": paper_id, "title": title, "abstract": abstract, "content": content,
                            "aspects": aspects, "review": review, "text": text}

                if not self.feature_expanding:
                    yield id_, raw_feature_info

                else:
                    additional_feature_info = get_features_sample_level_asap(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    # print(additional_feature_info)
                    yield id_, raw_feature_info
