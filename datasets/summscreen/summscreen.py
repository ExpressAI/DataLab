""" SummScreen: A Dataset for Abstractive Screenplay Summarization """
import json
import os
import datalabs
from datalabs import get_task, TaskType

_CITATION = """\
@article{DBLP:journals/corr/abs-2104-07091,
  author    = {Mingda Chen and
               Zewei Chu and
               Sam Wiseman and
               Kevin Gimpel},
  title     = {SummScreen: {A} Dataset for Abstractive Screenplay Summarization},
  journal   = {CoRR},
  volume    = {abs/2104.07091},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.07091},
  eprinttype = {arXiv},
  eprint    = {2104.07091},
  timestamp = {Mon, 19 Apr 2021 16:45:47 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-07091.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
SummScreen is a summarization dataset comprised of pairs of TV series transcripts and human written recaps.
The transcripts consist of dialogue utterances with speaker names, and descriptions of scenes or character actions. The recaps are human-written summaries of the corresponding transcripts.
see: https://arxiv.org/pdf/2104.07091.pdf
"""

_HOMEPAGE = "https://github.com/mingdachen/SummScreen"
_ABSTRACT = "summary"
_ARTICLE = "text"


def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download&confirm=t"
        
def restore(line):
    line = line.replace('@@ ', '')
    if line.endswith('@@'):
        line = line[:-2]
    return line

class SummScreenConfig(datalabs.BuilderConfig):
    """BuilderConfig for SummScreen."""

    def __init__(self, **kwargs):
        """BuilderConfig for SummScreen.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SummScreenConfig, self).__init__(**kwargs)


class SummScreenDataset(datalabs.GeneratorBasedBuilder):
    """SummScreen Dataset."""
    _FILE = _gdrive_url("1BvdIllGBo9d2-bzXQRzWuJXB04XPVmfF")
    BUILDER_CONFIGS = [
        SummScreenConfig(
            name="non-anonymized-ForeverDreaming",
            version=datalabs.Version("1.0.0"),
            description="SummScreen dataset for summarization, ForeverDreaming split, non-anonymized version, tokenized",
        ),
        SummScreenConfig(
            name="anonymized-ForeverDreaming",
            version=datalabs.Version("1.0.0"),
            description="SummScreen dataset for summarization, ForeverDreaming split, anonymized version, tokenized",
        ),
        SummScreenConfig(
            name="non-anonymized-TVMegaSite",
            version=datalabs.Version("1.0.0"),
            description="SummScreen dataset for summarization, TVMegaSite split, non-anonymized version, tokenized",
        ),
        SummScreenConfig(
            name="anonymized-TVMegaSite",
            version=datalabs.Version("1.0.0"),
            description="SummScreen dataset for summarization, TVMegaSite split, anonymized version, tokenized",
        ),
    ]
    DEFAULT_CONFIG_NAME = "non-anonymized-ForeverDreaming"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        f_path = dl_manager.download_and_extract(self._FILE)
        if self.config.name == "non-anonymized-ForeverDreaming":
            train_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_train.json")
            val_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_dev.json")
            test_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_test.json")
        elif self.config.name == "anonymized-ForeverDreaming":
            train_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_anonymize_train.json")
            val_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_anonymize_dev.json")
            test_path = os.path.join(f_path, "./SummScreen/ForeverDreaming/fd_anonymize_test.json")
        elif self.config.name == "non-anonymized-TVMegaSite":
            train_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_train.json")
            val_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_dev.json")
            test_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_test.json")
        else:
            train_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_anonymize_train.json")
            val_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_anonymize_dev.json")
            test_path = os.path.join(f_path, "./SummScreen/TVMegaSite/tms_anonymize_test.json")
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
        """Generate SummScreen examples."""
        with open(f_path, encoding="utf-8") as f:
            for (id_, x) in enumerate(f):
                data = json.loads(x)
                text = [restore(line) for line in data["Transcript"]]
                text = " ".join(text)
                summary = [restore(line) for line in data["Recap"]]
                summary = " ".join(summary)
                yield id_, {"text": text, "summary": summary}