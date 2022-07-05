import csv
import os
import datalabs
from datalabs import get_task, TaskType


_DESCRIPTION = """
 MReD consists of 7,089 meta-reviews and all its 45k meta-review sentences are manually annotated with one of the 9 carefully defined categories, including abstract, strength, decision, etc.
 See: https://aclanthology.org/2022.findings-acl.198.pdf
"""

_CITATION = """\
    @inproceedings{shen-etal-2022-mred,
    title = "{MR}e{D}: A Meta-Review Dataset for Structure-Controllable Text Generation",
    author = "Shen, Chenhui  and
      Cheng, Liying  and
      Zhou, Ran  and
      Bing, Lidong  and
      You, Yang  and
      Si, Luo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.198",
    doi = "10.18653/v1/2022.findings-acl.198",
    pages = "2521--2535",
    abstract = "When directly using existing text generation datasets for controllable generation, we are facing the problem of not having the domain knowledge and thus the aspects that could be controlled are limited. A typical example is when using CNN/Daily Mail dataset for controllable text summarization, there is no guided information on the emphasis of summary sentences. A more useful text generator should leverage both the input text and the control signal to guide the generation, which can only be built with deep understanding of the domain knowledge. Motivated by this vision, our paper introduces a new text generation dataset, named MReD. Our new dataset consists of 7,089 meta-reviews and all its 45k meta-review sentences are manually annotated with one of the 9 carefully defined categories, including abstract, strength, decision, etc. We present experimental results on start-of-the-art summarization models, and propose methods for structure-controlled generation with both extractive and abstractive models using our annotated data. By exploring various settings and analyzing the model behavior with respect to the control signal, we demonstrate the challenges of our proposed task and the values of our dataset MReD. Meanwhile, MReD also allows us to have a better understanding of the meta-review domain.",
}
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


class MReDConfig(datalabs.BuilderConfig):
    """BuilderConfig for MReD."""

    def __init__(self, **kwargs):
        """BuilderConfig for MReD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MReDConfig, self).__init__(**kwargs)


class MReDDataset(datalabs.GeneratorBasedBuilder):
    """MReD Dataset."""
    _URLs = {
        "concat": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/train_concat.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/test_concat.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/val_concat.csv",
        },
        "longest": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/train_longest.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/test_longest.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/val_longest.csv",
        },
        "merge": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/train_merge.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/test_merge.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/val_merge.csv",
        },
        "rate_concat": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/train_rate_concat.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/test_rate_concat.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/val_rate_concat.csv",
        },
        "rate_merge": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/train_rate_merge.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/test_rate_merge.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_uncontrolled_data/val_rate_merge.csv",
        },
        "concat_seg-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_concat_seg-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_concat_seg-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_concat_seg-ctrl.csv",
        },
        "longest_seg-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_longest_seg-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_longest_seg-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_longest_seg-ctrl.csv",
        },
        "merge_seg-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_merge_seg-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_merge_seg-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_merge_seg-ctrl.csv",
        },
        "rate_concat_seg-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_rate_concat_seg-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_rate_concat_seg-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_rate_concat_seg-ctrl.csv",
        },
        "rate_merge_seg-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_rate_merge_seg-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_rate_merge_seg-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_rate_merge_seg-ctrl.csv",
        },
        "concat_sent-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_concat_sent-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_concat_sent-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_concat_sent-ctrl.csv",
        },
        "longest_sent-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_longest_sent-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_longest_sent-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_longest_sent-ctrl.csv",
        },
        "merge_sent-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_merge_sent-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_merge_sent-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_merge_sent-ctrl.csv",
        },
        "rate_concat_sent-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_rate_concat_sent-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_rate_concat_sent-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_rate_concat_sent-ctrl.csv",
        },
        "rate_merge_sent-ctrl": {
            "train": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/train_rate_merge_sent-ctrl.csv",
            "test": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/test_rate_merge_sent-ctrl.csv",
            "val": "https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/filtered_controlled_data/val_rate_merge_sent-ctrl.csv",
        },
    }
    BUILDER_CONFIGS = [
        MReDConfig(
            name=k,
            version=datalabs.Version("1.0.0"),
            description=f"MReD dataset for summarization, {k} version",
        ) for k in _URLs.keys()
    ]
    DEFAULT_CONFIG_NAME = "concat"

    def _info(self):

        features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )

        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage="https://github.com/Shen-Chenhui/MReD/",
            citation=_CITATION,
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._URLs[self.config.name]["train"])
        test_path = dl_manager.download(self._URLs[self.config.name]["test"])
        val_path = dl_manager.download(self._URLs[self.config.name]["val"])
        
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate MReD examples."""
        cnt = 0
        # open the file as a csv file
        with open(f_path, "r") as f:
            reader = csv.reader(f)
            # skip the first line
            next(reader)
            for row in reader:
                yield cnt, {
                    _ARTICLE: row[0],
                    _ABSTRACT: row[1],
                }
                cnt += 1


