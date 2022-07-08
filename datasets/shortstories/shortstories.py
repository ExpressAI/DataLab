"""shortstories: Gold Corpus for Telegraphic Summarization"""
import os
import glob
import json
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@inproceedings{malireddy-etal-2018-gold,
    title = "Gold Corpus for Telegraphic Summarization",
    author = "Malireddy, Chanakya  and
      Somisetty, Srivenkata N M  and
      Shrivastava, Manish",
    booktitle = "Proceedings of the First Workshop on Linguistic Resources for Natural Language Processing",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-3810",
    pages = "71--77",
    abstract = "Most extractive summarization techniques operate by ranking all the source sentences and then select the top ranked sentences as the summary. Such methods are known to produce good summaries, especially when applied to news articles and scientific texts. However, they don{'}t fare so well when applied to texts such as fictional narratives, which don{'}t have a single central or recurrent theme. This is because usually the information or plot of the story is spread across several sentences. In this paper, we discuss a different summarization technique called Telegraphic Summarization. Here, we don{'}t select whole sentences, rather pick short segments of text spread across sentences, as the summary. We have tailored a set of guidelines to create such summaries and, using the same, annotate a gold corpus of 200 English short stories.",
}
"""

_DESCRIPTION = """\
We construct a corpus of 200 English short stories and their telegraphic summaries. 
50 abstractive summaries and 45 MCQs are also provided for evaluation purposes. 
see: https://aclanthology.org/W18-3810.pdf
"""

_HOMEPAGE = "https://github.com/m-chanakya/shortstories"
_ARTICLE = "text"
_ABSTRACT = "summary"


class ShortstoriesConfig(datalabs.BuilderConfig):
    """BuilderConfig for Shortstories."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for Shortstories.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ShortstoriesConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class ShortstoriesDataset(datalabs.GeneratorBasedBuilder):
    """Shortstories Dataset."""

    BUILDER_CONFIGS = [
        ShortstoriesConfig(
            name="abstractive",
            version=datalabs.Version("1.0.0"),
            description="Dataset for telegraphic summarization. Abstractive version",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        ShortstoriesConfig(
            name="extractive-resoomer",
            version=datalabs.Version("1.0.0"),
            description="Dataset for telegraphic summarization. Extractive version, annotator: resoomer",
            task_templates=[get_task(TaskType.extractive_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        ShortstoriesConfig(
            name="extractive-smmry",
            version=datalabs.Version("1.0.0"),
            description="Dataset for telegraphic summarization. Extractive version, annotator: smmry",
            task_templates=[get_task(TaskType.extractive_summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "abstractive"

    def _info(self):
        if "extractive" in self.config.name:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string"))
                }
            )
        else:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string")
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=["en"],
            task_templates=self.config.task_templates
        )

    def _split_generators(self, dl_manager):
        url = "https://github.com/m-chanakya/shortstories/archive/refs/heads/master.zip"
        f_path = dl_manager.download_and_extract(url)

        test_f_path = os.path.join(f_path, "shortstories-master")

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={"f_path": test_f_path}
            )
        ]

    def _generate_examples(self, f_path):
        """Generate shortstories examples."""
        name = self.config.name
        if "extractive" in name:
            type = name.split("-")[0].strip()
            person = name.split("-")[1].strip()
            files = glob.glob(os.path.join(f_path, type, person, "*.txt"))
        else:
            files = glob.glob(os.path.join(f_path, name, "*.txt"))

        datas = []
        for file in files:
            # summary
            f_summary = open(file, encoding="utf-8")
            summary_lines = f_summary.readlines()
            summary_sentences = []
            for summary_line in summary_lines:
                summary_line = summary_line.strip()
                if summary_line:
                    summary_sentences.append(summary_line)
            if "extractive" in name:
                summary = summary_sentences
            else:
                summary = " ".join(summary_sentences)

            # text
            file_name = os.path.basename(file)
            input_path = os.path.join(f_path, "stories", file_name)
            f_text = open(input_path, encoding="utf-8")
            text_lines = f_text.readlines()
            text_sentences = []
            for text_line in text_lines:
                text_line = text_line.strip()
                if text_line:
                    text_sentences.append(text_line)
            text = " ".join(text_sentences)
            datas.append((text, summary))

        for id_, (text, summary) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summary
            }
            yield id_, raw_feature_info
