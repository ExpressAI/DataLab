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
 The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. Linguists were asked to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger convesations. The style and register are diversified - conversations could be informal, semi-formal or formal, they may contain slang words, emoticons and typos. Then, the conversations were annotated with summaries. It was assumed that summaries should be a concise brief of what people talked about in the conversation in third person.
 The SAMSum dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes (non-commercial licence: CC BY-NC-ND 4.0).
 From paper: "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization" by B. Gliwa et al.
 See: https://aclanthology.org/D19-5409.pdf
"""
_CITATION = """\
    @inproceedings{gliwa-etal-2019-samsum,
    title = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization",
    author = "Gliwa, Bogdan  and
      Mochol, Iwona  and
      Biesek, Maciej  and
      Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5409",
    doi = "10.18653/v1/D19-5409",
    pages = "70--79",
    abstract = "This paper introduces the SAMSum Corpus, a new dataset with abstractive dialogue summaries. We investigate the challenges it poses for automated summarization by testing several models and comparing their results with those obtained on a corpus of news articles. We show that model-generated summaries of dialogues achieve higher ROUGE scores than the model-generated summaries of news {--} in contrast with human evaluators{'} judgement. This suggests that a challenging task of abstractive dialogue summarization requires dedicated models and non-standard quality measures. To our knowledge, our study is the first attempt to introduce a high-quality chat-dialogues corpus, manually annotated with abstractive summarizations, which can be used by the research community for further studies.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"

class SAMSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for SAMSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for SAMSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SAMSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates



class SAMSumDataset(datalabs.GeneratorBasedBuilder):
    """SAMSum Dataset."""
    _FILE_ID = "1Wq9A5ZXOMZN3w3HVjIGwBz25XVbbgNr5"
    BUILDER_CONFIGS = [
        SAMSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="SAMSum dataset for summarization, single document version",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        SAMSumConfig(
            name="dialogue",
            version=datalabs.Version("1.0.0"),
            description="SAMSum dataset for summarization, dialogue summarization version",
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
            homepage=None,
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))
        train_path = os.path.join(f_path, "train.json")
        test_path = os.path.join(f_path, "test.json")
        val_path = os.path.join(f_path, "val.json")
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate SAMSum examples."""
        with open(f_path, encoding="utf-8") as f:
            data = json.load(f)
        for (id_, article) in enumerate(data):

            # if id_ > 100:
            #     break

            if "document" in self.config.name:

                raw_feature_info = {
                    _ARTICLE: article["dialogue"],
                    _ABSTRACT: article["summary"]
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
                text = [x.strip() for x in article["dialogue"].split("\n")]
                text = [x for x in text if len(x) > 0]
                for x in text:
                    speaker, content = x.split(":", 1)
                    dialogue.append({"speaker": speaker, "text": content})
                yield id_, {
                    "dialogue": dialogue,
                    _ABSTRACT: [article["summary"]],
                }