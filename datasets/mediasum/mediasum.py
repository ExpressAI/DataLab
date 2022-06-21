"""MediaSum: Large-scale Media Interview Summarization Dataset"""
import json
import os
import datalabs
from datalabs import get_task, TaskType

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
)
from datalabs.utils.more_features import (
    get_feature_schemas,
)

_DESCRIPTION = """
MediaSum is a large-scale media interview dataset contains 463.6K transcripts with abstractive summaries, 
collected from interview transcripts and overview / topic descriptions from NPR and CNN.
We obtain this dataset from https://huggingface.co/datasets/ccdv/mediasum.
See: https://aclanthology.org/2021.naacl-main.474.pdf
"""

_CITATION = """\
@inproceedings{zhu-etal-2021-mediasum,
    title = "{M}edia{S}um: A Large-scale Media Interview Dataset for Dialogue Summarization",
    author = "Zhu, Chenguang  and
      Liu, Yang  and
      Mei, Jie  and
      Zeng, Michael",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.474",
    doi = "10.18653/v1/2021.naacl-main.474",
    pages = "5927--5934"
}
"""
_HOMEPAGE = "https://github.com/zcgzcgzcg1/MediaSum"
_ABSTRACT = "summary"
_ARTICLE = "text"


class MediaSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for MediaSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for MediaSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MediaSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class MediaSumConfigDataset(datalabs.GeneratorBasedBuilder):
    """MediaSumConfig Dataset."""
    BUILDER_CONFIGS = [
        MediaSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="MediaSum dataset for summarization, single document summarization version",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        MediaSumConfig(
            name="dialogue",
            version=datalabs.Version("1.0.0"),
            description="MediaSum dataset for summarization, dialogue summarization version",
            task_templates=[get_task(TaskType.dialog_summarization)(
                source_column="dialogue",
                reference_column=_ABSTRACT)]
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
                _ABSTRACT: datalabs.Value("string"),
            })
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            languages=["en"],
            citation=_CITATION,
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        train_url = "https://huggingface.co/datasets/ccdv/mediasum/resolve/main/train_data.zip"
        valid_url = "https://huggingface.co/datasets/ccdv/mediasum/resolve/main/val_data.zip"
        test_url = "https://huggingface.co/datasets/ccdv/mediasum/resolve/main/test_data.zip"

        train_f_path = dl_manager.download_and_extract(train_url)
        valid_f_path = dl_manager.download_and_extract(valid_url)
        test_f_path = dl_manager.download_and_extract(test_url)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": os.path.join(train_f_path,"train_data.txt")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": os.path.join(valid_f_path,"val_data.txt")},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": os.path.join(test_f_path,"test_data.txt")},
            )
        ]

    def _generate_examples(self, f_path):
        """Generate MediaSum examples."""
        datas = []
        original_datas = []
        with open(f_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                original_datas.append(json.loads(line))

        for original_data in original_datas:
            summary = original_data["summary"].strip()
            speakers = original_data["speaker"]
            utterances = original_data["utt"]
            datas.append({"summary": summary, "speakers": speakers, "utterances": utterances})

        for id_, data in enumerate(datas):
            if "document" in self.config.name:
                text = ""
                for speaker, utterance in zip(data["speakers"], data["utterances"]):
                    text = text + speaker + " : " + utterance + " "

                raw_feature_info = {
                    _ARTICLE: text.strip(),
                    _ABSTRACT: data["summary"],
                }

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
            else:
                dialogue = []
                for speaker, utterance in zip(data["speakers"], data["utterances"]):
                    dialogue.append({"speaker": speaker, "text": utterance})

                yield id_, {
                    "dialogue": dialogue,
                    _ABSTRACT: data["summary"],
                }
