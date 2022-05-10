import json
import os
import datalabs
from datalabs.tasks import Summarization, DialogSummarization

# the following package are needed when more additional features are expected to be calculated
from datalabs.operations.featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
    )
from datalabs.utils.more_features import (
    get_feature_schemas,
)


_DESCRIPTION = """
Critical Role Dungeons and Dragons Dataset (CRD3) is based on Critical Role, an unscripted, live-streamed show where a fixed group of people play Dungeons and Dragons, an openended role-playing game. 
The dataset is collected from 159 Critical Role episodes transcribed to text dialogues, consisting of 398,682 turns. 
It also includes corresponding abstractive summaries collected from the Fandom wiki. 
The dataset is linguistically unique in that the narratives are generated entirely through player collaboration and spoken interaction.
See: https://aclanthology.org/2020.acl-main.459.pdf
"""
_CITATION = """\
    @inproceedings{rameshkumar-bailey-2020-storytelling,
    title = "Storytelling with Dialogue: {A} {Critical Role Dungeons and Dragons Dataset}",
    author = "Rameshkumar, Revanth  and
      Bailey, Peter",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.459",
    doi = "10.18653/v1/2020.acl-main.459",
    pages = "5121--5134",
    abstract = "This paper describes the Critical Role Dungeons and Dragons Dataset (CRD3) and related analyses. Critical Role is an unscripted, live-streamed show where a fixed group of people play Dungeons and Dragons, an open-ended role-playing game. The dataset is collected from 159 Critical Role episodes transcribed to text dialogues, consisting of 398,682 turns. It also includes corresponding abstractive summaries collected from the Fandom wiki. The dataset is linguistically unique in that the narratives are generated entirely through player collaboration and spoken interaction. For each dialogue, there are a large number of turns, multiple abstractive summaries with varying levels of detail, and semantic ties to the previous dialogues. In addition, we provide a data augmentation method that produces 34,243 summary-dialogue chunk pairs to support current neural ML approaches, and we provide an abstractive summarization benchmark and evaluation.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"

_BASE_URL = "https://raw.githubusercontent.com/RevanthRameshkumar/CRD3/master/baseline/data/aligned data/"


class CRD3Config(datalabs.BuilderConfig):
    """BuilderConfig for CRD3."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for CRD3.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CRD3Config, self).__init__(**kwargs)
        self.task_templates = task_templates



class CRD3Dataset(datalabs.GeneratorBasedBuilder):
    """CRD3 Dataset."""
    BUILDER_CONFIGS = [
        CRD3Config(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="CRD3 dataset for summarization, single document version",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        CRD3Config(
            name="dialogue",
            version=datalabs.Version("1.0.0"),
            description="CRD3 dataset for summarization, dialogue summarization version",
            task_templates=[DialogSummarization(
                text_column="dialogue", summary_column=_ABSTRACT)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

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
                features_sample, features_dataset = get_feature_schemas(features_sample, get_schema_of_sample_level_features)

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
            homepage="https://github.com/RevanthRameshkumar/CRD3",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        test_file_id = dl_manager.download(os.path.join(_BASE_URL, "test_files"))
        train_file_id = dl_manager.download(os.path.join(_BASE_URL, "train_files"))
        val_file_id = dl_manager.download(os.path.join(_BASE_URL, "val_files"))
        file_ids = set()
        with open(train_file_id) as f:
            file_ids.update([line.strip() for line in f])
        with open(val_file_id) as f:
            file_ids.update([line.strip() for line in f])
        with open(test_file_id) as f:
            file_ids.update([line.strip() for line in f])

        skip_id_1 = [15, 26, 41, 45, 48, 59, 66, 67, 69, 70, 74, 
                     77, 78, 79, 80, 81, 82, 83, 84]
        urls = []
        url_base = os.path.join(_BASE_URL, "c=2")
        for i in range(1, 116):
            if f"C1E{i:03d}" in file_ids and i not in skip_id_1:
                urls.append(os.path.join(url_base, f"C1E{i:03d}_2_0.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_2_1.json"))
        for i in range(1, 47):
            if f"C2E{i:03d}" in file_ids:
                urls.append(os.path.join(url_base, f"C2E{i:03d}_2_0.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_2_1.json"))

        skip_id_1 = [15, 16, 26, 41, 44, 45, 48, 54, 55, 59, 60, 62, 66, 67, 69, 70, 74, 
                     77, 78, 79, 80, 81, 82, 83, 84]
        url_base = os.path.join(_BASE_URL, "c=3")
        for i in range(1, 116):
            if f"C1E{i:03d}" in file_ids and i not in skip_id_1:
                urls.append(os.path.join(url_base, f"C1E{i:03d}_3_0.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_3_1.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_3_2.json"))
        for i in range(1, 47):
            if f"C2E{i:03d}" in file_ids:
                urls.append(os.path.join(url_base, f"C2E{i:03d}_3_0.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_3_1.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_3_2.json"))

        skip_id_1 = [15, 16, 26, 41, 44, 45, 46, 48, 54, 55, 58, 59, 60, 62, 66, 67, 69, 70, 74, 
                     77, 78, 79, 80, 81, 82, 83, 84]
        url_base = os.path.join(_BASE_URL, "c=4")
        for i in range(1, 116):
            if f"C1E{i:03d}" in file_ids and i not in skip_id_1:
                urls.append(os.path.join(url_base, f"C1E{i:03d}_4_0.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_4_1.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_4_2.json"))
                urls.append(os.path.join(url_base, f"C1E{i:03d}_4_3.json"))
        for i in range(1, 47):
            if f"C2E{i:03d}" in file_ids:
                urls.append(os.path.join(url_base, f"C2E{i:03d}_4_0.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_4_1.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_4_2.json"))
                urls.append(os.path.join(url_base, f"C2E{i:03d}_4_3.json"))

        json_files = {url: dl_manager.download(url) for url in urls}

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_paths": json_files, "f_ids": train_file_id}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_paths": json_files, "f_ids": val_file_id}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_paths": json_files, "f_ids": test_file_id}
            ),
        ]

    def _generate_examples(self, f_paths, f_ids):
        """Generate CRD3 examples."""
        cnt = 0
        with open(f_ids) as f:
            files = set([line.strip() for line in f])
        for f_path in f_paths:
            file_id = f_path.split("/")[-1].split("_")[0]
            if file_id in files:
                with open(f_paths[f_path]) as f:
                    data = json.load(f)
                for x in data:
                    summary = x["CHUNK"]
                    if "document" in self.config.name:
                        text = []
                        for turn in x["TURNS"]:
                            text.append("{} {}".format(" ".join(turn["NAMES"]), " ".join(turn["UTTERANCES"])))
                        raw_feature_info = {
                            _ARTICLE: text,
                            _ABSTRACT: summary
                        }

                        if not self.feature_expanding:
                            yield cnt, raw_feature_info
                        else:
                            additional_feature_info = get_features_sample_level(raw_feature_info)
                            raw_feature_info.update(additional_feature_info)
                            yield cnt, raw_feature_info
                        cnt += 1

                    else:
                        dialogue = []
                        for turn in x["TURNS"]:
                            dialogue.append({"speaker": " ".join(turn["NAMES"]), "text": " ".join(turn["UTTERANCES"])})
                        yield cnt, {
                            "dialogue": dialogue,
                            _ABSTRACT: [summary],
                        }
                        cnt += 1