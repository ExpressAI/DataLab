"""AmaSum: abstractive opinion summarization dataset."""
import json
import os

import datalabs
from datalabs import get_task, TaskType
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN

_CITATION = """\
@inproceedings{brazinskas-etal-2021-learning,
    title = "Learning Opinion Summarizers by Selecting Informative Reviews",
    author = "Bra{\v{z}}inskas, Arthur  and
      Lapata, Mirella  and
      Titov, Ivan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.743",
    doi = "10.18653/v1/2021.emnlp-main.743",
    pages = "9424--9442",
    abstract = "Opinion summarization has been traditionally approached with unsupervised, weakly-supervised and few-shot learning techniques. In this work, we collect a large dataset of summaries paired with user reviews for over 31,000 products, enabling supervised training. However, the number of reviews per product is large (320 on average), making summarization {--} and especially training a summarizer {--} impractical. Moreover, the content of many reviews is not reflected in the human-written summaries, and, thus, the summarizer trained on random review subsets hallucinates. In order to deal with both of these challenges, we formulate the task as jointly learning to select informative subsets of reviews and summarizing the opinions expressed in these subsets. The choice of the review subset is treated as a latent variable, predicted by a small and simple selector. The subset is then fed into a more powerful summarizer. For joint training, we use amortized variational inference and policy gradient methods. Our experiments demonstrate the importance of selecting informative reviews resulting in improved quality of summaries and reduced hallucinations.",
}
"""

_DESCRIPTION = """\
We collect a large dataset of summaries paired with user reviews for over 31,000 products.
see: https://aclanthology.org/2021.emnlp-main.743.pdf
"""

_HOMEPAGE = "https://github.com/abrazinskas/SelSum"
_LICENSE = "Used for non-commercial and educational purposes (https://github.com/abrazinskas/SelSum/blob/master/data/LICENSE.txt)"
_ARTICLE = "text"
_ABSTRACT = "summary"


class AmaSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for AmaSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for AmaSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AmaSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class AmaSumDataset(datalabs.GeneratorBasedBuilder):
    """AmaSum Dataset."""

    _FILE_URL = "https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/raw_min_10_max_100_revs.zip"

    BUILDER_CONFIGS = [
        AmaSumConfig(
            name="verdict-ref",
            version=datalabs.Version("1.0.0"),
            description="Abstractive opinion summarization dataset, verdict reference version.",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
        AmaSumConfig(
            name="pros-ref",
            version=datalabs.Version("1.0.0"),
            description="Abstractive opinion summarization dataset, pros reference version.",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
        AmaSumConfig(
            name="cons-ref",
            version=datalabs.Version("1.0.0"),
            description="Abstractive opinion summarization dataset, cons reference version.",
            task_templates=[
                get_task(TaskType.multi_doc_summarization)(
                    source_column=_MDS_TEXT_COLUMN, reference_column=_ABSTRACT
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "verdict-ref"

    def _info(self):
        features_sample = datalabs.Features(
            {
                _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                _ABSTRACT: datalabs.Value("string"),
            }
        )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(self._FILE_URL)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path, "min_10_max_100_revs_filt_complete/train"
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path, "min_10_max_100_revs_filt_complete/valid"
                    )
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "f_path": os.path.join(
                        f_path, "min_10_max_100_revs_filt_complete/test"
                    )
                },
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate AmaSum examples."""
        files = os.listdir(f_path)
        datas = []
        for file in files:
            f = open(os.path.join(f_path, file), encoding="utf-8")
            data = json.load(f)

            texts = [review["text"] for review in data["customer_reviews"]]  # list

            verdict_summaries = data["website_summaries"][0]["verdict"]  # str
            pros_summaries = " ".join(data["website_summaries"][0]["pros"])  # str
            cons_summaries = " ".join(data["website_summaries"][0]["cons"])  # str

            datas.append(
                {
                    "texts": texts,
                    "verdict": verdict_summaries,
                    "cons": cons_summaries,
                    "pros": pros_summaries,
                }
            )

        for id_, data in enumerate(datas):

            if self.config.name == "verdict-ref":
                summary = data["verdict"]
            elif self.config.name == "pros-ref":
                summary = data["pros"]
            elif self.config.name == "cons-ref":
                summary = data["cons"]

            raw_feature_info = {_MDS_TEXT_COLUMN: data["texts"], _ABSTRACT: summary}
            yield id_, raw_feature_info
