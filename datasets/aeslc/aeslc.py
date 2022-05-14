"""AESLC: Annotated Enron Subject Line Corpus."""
import os
import datalabs
from datalabs import get_task, TaskType



_CITATION = """\
@inproceedings{zhang-tetreault-2019-email,
    title = "This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation",
    author = "Zhang, Rui  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1043",
    doi = "10.18653/v1/P19-1043",
    pages = "446--456"
}
"""

_DESCRIPTION = """\
To introduce the task, we build the first dataset, Annotated Enron Subject Line Corpus (AESLC), 
by leveraging the Enron Corpus and crowdsourcing.
see: https://aclanthology.org/P19-1043.pdf
"""

_HOMEPAGE = "https://github.com/ryanzhumich/AESLC"
_ARTICLE = "text"
_ABSTRACT = "summaries"
_LANGUAGES = ["en"]
_TASK_TEMPLATES = [
    get_task(TaskType.multi_ref_summarization)(
        source_column=_ARTICLE,
        reference_column=_ABSTRACT)
]

class AESLCConfig(datalabs.BuilderConfig):
    """BuilderConfig for AESLC."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for AESLC.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AESLCConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class AESLCDataset(datalabs.GeneratorBasedBuilder):
    """AESLC Dataset."""

    BUILDER_CONFIGS = [
        AESLCConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="Email subject line generation Dataset.",
            task_templates=_TASK_TEMPLATES
        )
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Sequence(datalabs.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            version=self.VERSION,
            languages=_LANGUAGES,
            task_templates=_TASK_TEMPLATES,
        )

    def _split_generators(self, dl_manager):
        url = "https://github.com/ryanzhumich/AESLC/archive/refs/heads/master.zip"
        f_path = dl_manager.download_and_extract(url)

        train_dir = os.path.join(f_path, "AESLC-master/enron_subject_line/train")
        train_f_paths = os.listdir(train_dir)
        train_f_paths = [os.path.join(f_path, "AESLC-master/enron_subject_line/train", train_f_path) for train_f_path in
                         train_f_paths]

        dev_dir = os.path.join(f_path, "AESLC-master/enron_subject_line/dev")
        dev_f_paths = os.listdir(dev_dir)
        dev_f_paths = [os.path.join(f_path, "AESLC-master/enron_subject_line/dev", dev_f_path) for dev_f_path in
                       dev_f_paths]

        test_dir = os.path.join(f_path, "AESLC-master/enron_subject_line/test")
        test_f_paths = os.listdir(test_dir)
        test_f_paths = [os.path.join(f_path, "AESLC-master/enron_subject_line/test", test_f_path) for test_f_path in
                        test_f_paths]

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_paths": train_f_paths}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_paths": dev_f_paths}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_paths": test_f_paths}
            ),
        ]

    def _generate_examples(self, f_paths):
        """Generate AESLC examples."""
        datas = []
        for f_path in f_paths:
            if ".subject" in f_path:
                f = open(f_path, encoding="utf-8")
                lines = f.readlines()

                if "/train" in f_path:
                    sentences = [line.strip() for line in lines[:-2]]
                    text = " ".join(sentences)
                    summaries = [lines[-1].strip()]
                    datas.append((text, summaries))
                else:
                    sentences = [line.strip() for line in lines[:-11]]
                    text = " ".join(sentences)
                    summaries = [lines[-10].strip(), lines[-7].strip(), lines[-4].strip(), lines[-1].strip()]
                    datas.append((text, summaries))

        for id_, (text, summaries) in enumerate(datas):
            raw_feature_info = {
                _ARTICLE: text,
                _ABSTRACT: summaries
            }
            yield id_, raw_feature_info
