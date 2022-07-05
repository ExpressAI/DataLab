"""The AMI Meeting Corpus: A pre-announcement."""
import json
import os

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
)

import datalabs
from datalabs import get_task, TaskType
from datalabs.utils.more_features import get_feature_schemas

_CITATION = """\
@inproceedings{Carletta2005TheAM,
  title={The AMI Meeting Corpus: A Pre-announcement},
  author={Jean Carletta and Simone Ashby and Sebastien Bourban and Mike Flynn and Ma{\"e}l Guillemot and Thomas Hain and Jaroslav Kadlec and Vasilis Karaiskos and Wessel Kraaij and Melissa Kronenthal and Guillaume Lathoud and Mike Lincoln and Agnes Lisowska Masson and Iain McCowan and Wilfried Post and Dennis Reidsma and Pierre D. Wellner},
  booktitle={MLMI},
  year={2005}
}
"""

_DESCRIPTION = """\
The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting recordings.
The AMI Meeting Corpus is widely used in meeting summarization.
This is the preprocessed version provided by https://github.com/microsoft/HMNet.
"""

_HOMEPAGE = "https://groups.inf.ed.ac.uk/ami/corpus/"
_LICENSE = "MIT License"
_ARTICLE = "text"
_ABSTRACT = "summary"


def get_article_summary_urls(mode):
    urls = []
    mode2num = {"train": 32, "dev": 20, "test": 20}
    root_url = "https://raw.githubusercontent.com/microsoft/HMNet/main/ExampleRawData/meeting_summarization/AMI_proprec/{}".format(
        mode
    )
    for index in range(0, mode2num[mode]):
        url = os.path.join(root_url, "split_{}.jsonl.gz".format(str(index)))
        urls.append(url)
    return urls


class AMIConfig(datalabs.BuilderConfig):
    """BuilderConfig for AMIConfig."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for AMIConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AMIConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class AMIDataset(datalabs.GeneratorBasedBuilder):
    """AMI Dataset."""

    BUILDER_CONFIGS = [
        AMIConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="AMI dataset for summarization, single document version",
            task_templates=[
                get_task(TaskType.summarization)(
                    source_column=_ARTICLE, reference_column=_ABSTRACT
                )
            ],
        ),
        AMIConfig(
            name="dialogue",
            version=datalabs.Version("1.0.0"),
            description="AMI dataset for summarization, dialogue summarization version",
            task_templates=[
                get_task(TaskType.dialog_summarization)(
                    source_column="dialogue", reference_column=_ABSTRACT
                )
            ],
        ),
    ]
    DEFAULT_CONFIG_NAME = "dialogue"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_dataset = {}

        if "document" in self.config.name:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(
                    features_sample, get_schema_of_sample_level_features
                )

        else:
            features_sample = datalabs.Features(
                {
                    "dialogue": datalabs.Sequence(
                        datalabs.Features(
                            {
                                "speaker": datalabs.Value("string"),
                                "text": datalabs.Value("string"),
                            }
                        )
                    ),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            languages=["en"],
            license=_LICENSE,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        train_urls = get_article_summary_urls("train")
        valid_urls = get_article_summary_urls("dev")
        test_urls = get_article_summary_urls("test")

        # This is a `safe` method to download files, since multiprocessing failed in some devices.
        # Will be uploaded to the multiprocessing in the future.
        train_f_paths = []
        for train_url in train_urls:
            train_f_path = dl_manager.download_and_extract(train_url)
            train_f_paths.append(train_f_path)

        valid_f_paths = []
        for valid_url in valid_urls:
            valid_f_path = dl_manager.download_and_extract(valid_url)
            valid_f_paths.append(valid_f_path)

        test_f_paths = []
        for test_url in test_urls:
            test_f_path = dl_manager.download_and_extract(test_url)
            test_f_paths.append(test_f_path)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_paths": train_f_paths}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_paths": valid_f_paths}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_paths": test_f_paths}
            ),
        ]

    def _generate_examples(self, f_paths):
        """Generate AMI examples."""
        original_datas = []
        for f_path in f_paths:
            f = open(f_path, encoding="utf-8")
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    original_data = json.loads(line)
                    original_datas.append(original_data)

        datas = []
        for original_data in original_datas:
            dialogue = []
            summary = " ".join(original_data["summary"]).strip()
            transcripts = original_data["meeting"]
            for transcript in transcripts:
                role = transcript["role"].strip()
                utterance = " ".join(transcript["utt"]["word"]).strip()
                dialogue.append({"speaker": role, "text": utterance})
            datas.append({"dialogue": dialogue, "summary": summary})

        for (id_, data) in enumerate(datas):

            if "document" in self.config.name:
                text = ""
                for one in data["dialogue"]:
                    text = text + one["speaker"] + " : " + one["text"] + " "
                text = text.strip()

                raw_feature_info = {_ARTICLE: text, _ABSTRACT: data["summary"]}

                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(
                        raw_feature_info
                    )
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info

            else:
                yield id_, {
                    "dialogue": data["dialogue"],
                    _ABSTRACT: data["summary"],
                }
