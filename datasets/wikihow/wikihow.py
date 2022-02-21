import json
import os
import datalabs
from datalabs.tasks import Summarization
import tempfile
import subprocess

_DESCRIPTION = """
 WikiHow is a new large-scale dataset using the online WikiHow (http://www.wikihow.com/) knowledge base.
 Each article consists of multiple paragraphs and each paragraph starts with a sentence summarizing it. By merging the paragraphs to form the article and the paragraph outlines to form the summary, the resulting version of the dataset contains more than 200,000 long-sequence pairs.
 From paper: "WikiHow: A Large Scale Text Summarization Dataset" by M. Koupaee et al.
 See: https://arxiv.org/pdf/1810.09305.pdf
"""
_CITATION = """\
    @article{DBLP:journals/corr/abs-1810-09305,
  author    = {Mahnaz Koupaee and
               William Yang Wang},
  title     = {WikiHow: {A} Large Scale Text Summarization Dataset},
  journal   = {CoRR},
  volume    = {abs/1810.09305},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.09305},
  eprinttype = {arXiv},
  eprint    = {1810.09305},
  timestamp = {Wed, 31 Oct 2018 14:24:29 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-09305.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"

def _gdrive_url(id):
    return f"https://drive.google.com/uc?id={id}&export=download"

def custom_download(url, path):
    with tempfile.TemporaryDirectory() as tmpdir:
        response = subprocess.check_output([
            "wget", "--save-cookies", os.path.join(tmpdir, "cookies.txt"), 
            f"{url}", "-O-"])
        with open(os.path.join(tmpdir, "response.txt"), "w") as f:
            f.write(response.decode("utf-8"))
        response = subprocess.check_output(["sed", "-rn", 's/.*confirm=([0-9A-Za-z_]+).*/\\1/p', os.path.join(tmpdir, "response.txt")])
        response = response.decode("utf-8")
        subprocess.check_output([
            "wget", "--load-cookies", os.path.join(tmpdir, "cookies.txt"), "-O", path,
            url+f"&confirm={response}"])

class WikiHowConfig(datalabs.BuilderConfig):
    """BuilderConfig for WikiHow."""

    def __init__(self, **kwargs):
        """BuilderConfig for WikiHow.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiHowConfig, self).__init__(**kwargs)


class WikiHowDataset(datalabs.GeneratorBasedBuilder):
    """WikiHow Dataset."""
    _FILE_ID = "1n6RQIZBGkCloxh6dSHAaDC8XsDBn2V_u"
    BUILDER_CONFIGS = [
        WikiHowConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="WikiHow dataset for summarization, document",
        ),
    ]
    DEFAULT_CONFIG_NAME = "document"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    # "id": datalab.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=None,
            citation=_CITATION,
            task_templates=[Summarization(
                text_column=_ARTICLE,
                summary_column=_ABSTRACT),
            ],
        )

    def _split_generators(self, dl_manager):
        # f_path = dl_manager.download_and_extract(_gdrive_url(self._FILE_ID))
        f_path = dl_manager.download_custom(_gdrive_url(self._FILE_ID), custom_download)
        f_path = dl_manager.extract(f_path)
        train_path = os.path.join(f_path, "train.jsonl")
        test_path = os.path.join(f_path, "test.jsonl")
        val_path = os.path.join(f_path, "val.jsonl")
        

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
        """Generate WikiHow examples."""
        with open(f_path, encoding="utf-8") as f:
            for (id_, line) in enumerate(f):
                data = json.loads(line)
                yield id_, {
                    _ARTICLE: data["article"],
                    _ABSTRACT: data["summary"],
                }