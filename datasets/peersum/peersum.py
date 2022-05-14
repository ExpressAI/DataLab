"""PeerSum: A Peer Review Dataset for Abstractive Multi-document Summarization."""
import os
import json
import datalabs
from datalabs import get_task, TaskType



_CITATION = """\
@article{Li2022PeerSumAP,
  title={PeerSum: A Peer Review Dataset for Abstractive Multi-document Summarization},
  author={Miao Li and Jianzhong Qi and Jey Han Lau},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.01769}
}
"""

_DESCRIPTION = """\
PeerSum is a multi-document summarization dataset, which is constructed based on peer reviews in the Openreview system. 
This dataset differs from other MDS datasets (e.g., Multi-News, WCEP, WikiSum, and Multi-XScience) in that our summaries (i.e., the metareviews) are 
highly abstractive and they are real summaries of the source documents (i.e., the reviews) and 
it also features disagreements among source documents. 
see: https://arxiv.org/pdf/2203.01769.pdf
"""

_HOMEPAGE = "https://github.com/oaimli/PeerSum"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"


class PeerSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for PeerSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for PeerSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PeerSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class PeerSumDataset(datalabs.GeneratorBasedBuilder):
    """PeerSum Dataset."""
    _FILE_ID = "13et-nzcHrNg5sZRZs_mP2l32wQ0WBBbT"

    BUILDER_CONFIGS = [
        PeerSumConfig(
            name="document",
            version=datalabs.Version("2.0.0"),
            description="A peer review dataset for abstractive multi-document summarization, single document version.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        PeerSumConfig(
            name="dialogue",
            version=datalabs.Version("2.0.0"),
            description="A peer review dataset for abstractive multi-document summarization, multi-document document version.",
            task_templates=[get_task(TaskType.dialog_summarization)(
                source_column="dialogue",
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_dataset = {}

        if self.config.name == "document":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        elif self.config.name == "dialogue":
            features_sample = datalabs.Features(
                {
                    "dialogue": datalabs.Sequence(datalabs.Features({
                        "speaker": datalabs.Value("string"),
                        "text": datalabs.Value("string")
                    })),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download(_gdrive_url(self._FILE_ID))

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": f_path, "split": "train"}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": f_path, "split": "val"}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": f_path, "split": "test"}
            ),
        ]

    def _generate_examples(self, f_path, split):
        with open(f_path, encoding="utf-8") as f:
            original_datas = json.load(f)

        datas = []
        for original_data in original_datas:
            label = original_data["label"]
            if label != split:
                continue

            reviews = original_data["reviews"]

            texts = []
            speakers = []
            for review in reviews:
                if "writer" in review.keys():
                    texts.append(review["content"]["comment"].strip().replace("\n", ""))
                    speakers.append(review["writer"].strip())

            summary = original_data["meta_review"]
            datas.append((speakers, texts, summary))

        if self.config.name == "document":
            for id_, (speakers, texts, summary) in enumerate(datas):
                input = ""
                for speaker, text in zip(speakers, texts):
                    input = input + speaker + " : " + text + " "

                raw_feature_info = {
                    _ARTICLE: input,
                    _ABSTRACT: summary
                }

                yield id_, raw_feature_info

        elif self.config.name == "dialogue":
            for id_, (speakers, texts, summary) in enumerate(datas):
                dialogue = []
                for speaker, text in zip(speakers, texts):
                    dialogue.append({"speaker": speaker, "text": text})

                raw_feature_info = {
                    "dialogue": dialogue,
                    _ABSTRACT: summary
                }
                yield id_, raw_feature_info
