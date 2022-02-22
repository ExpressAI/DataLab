import json
import os
import datalabs
from datalabs.tasks import Summarization, DialogSummarization

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
    )
from datalabs.utils.more_features import (
    get_feature_schemas,
)



_DESCRIPTION = """
 DialogSum contains face-to-face spoken dialogues that cover a wide range of daily-life topics, including schooling, work, medication, shopping, leisure, travel.
 See: https://aclanthology.org/2020.emnlp-main.648/
 See: https://github.com/cylnlp/dialogsum
"""
_CITATION = """\
    @inproceedings{chen-etal-2021-dialogsum,
    title = "{D}ialog{S}um: {A} Real-Life Scenario Dialogue Summarization Dataset",
    author = "Chen, Yulong  and
      Liu, Yang  and
      Chen, Liang  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.449",
    doi = "10.18653/v1/2021.findings-acl.449",
    pages = "5062--5074",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"


class DialogSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for DialogSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for DialogSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DialogSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class DialogSumDataset(datalabs.GeneratorBasedBuilder):
    """DialogSum Dataset."""
    _TRAIN_URL = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.train.jsonl"
    _VAL_URL = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.dev.jsonl"
    _TEST_URL = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.test.jsonl"
    BUILDER_CONFIGS = [
        DialogSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="DialogSum dataset for summarization, single document summarization version",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        DialogSumConfig(
            name="dialogue",
            version=datalabs.Version("1.0.0"),
            description="DialogSum dataset for summarization, dialogue summarization version",
            task_templates=[DialogSummarization(
                text_column="dialogue", summary_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "dialogue"

    def _info(self):
        features_dataset = {}
        # Should return a datalab.DatasetInfo object
        if "document" in self.config.name:


            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                        get_schema_of_sample_level_features)


        else:
            features_sample = datalabs.Features({
                "dialogue": datalabs.Sequence(datalabs.Features({
                        "speaker": datalabs.Value("string"),
                        "text": datalabs.Value("string")
                        })),
                _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
            })
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage="https://github.com/cylnlp/dialogsum",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._TRAIN_URL)
        val_path = dl_manager.download(self._VAL_URL)
        test_path = dl_manager.download(self._TEST_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path, "split": "train"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path, "split": "val"},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path, "split": "test"},
            ),
        ]

    def _generate_examples(self, f_path, split):
        """Generate DialogSum examples."""
        with open(f_path) as f:
            for (id_, x) in enumerate(f):
                x = json.loads(x)
                if split == "test":
                    if "document" in self.config.name:
                        summary = x["summary1"] # only keep the first summary
                    else:
                        summary = [x[f"summary{i}"] for i in range(1, 4)]
                else:
                    if "document" in self.config.name:
                        summary = x["summary"]
                    else:
                        summary = [x["summary"]]  # make it a list
                text = x["dialogue"]
                if "document" in self.config.name:

                    raw_feature_info = {
                        _ARTICLE: text,
                        _ABSTRACT: summary,
                    }

                    if not self.feature_expanding:
                        yield id_, raw_feature_info
                    else:
                        additional_feature_info = get_features_sample_level(raw_feature_info)
                        raw_feature_info.update(additional_feature_info)
                        # print(additional_feature_info)
                        yield id_, raw_feature_info


                else:
                    dialogue = []
                    text = [x.strip() for x in text.split("\n")]
                    for x in text:
                        speaker, content = x.split(":", 1)
                        dialogue.append({"speaker": speaker, "text": content})
                    yield id_, {
                        "dialogue": dialogue,
                        _ABSTRACT: summary,
                    }
