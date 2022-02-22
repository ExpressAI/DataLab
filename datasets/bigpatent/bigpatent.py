import json
import os
import datalabs
from datalabs.tasks import Summarization
import gzip
import tempfile
import subprocess

_DESCRIPTION = """
 BIGPATENT consists of 1.3 million records of U.S. patent documents along with human written abstractive summaries.
 From paper: "BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization" by B. Gliwa et al.
 See: https://aclanthology.org/P19-1212.pdf
 See: https://evasharma.github.io/bigpatent/
"""
_CITATION = """\
    @inproceedings{sharma-etal-2019-bigpatent,
    title = "{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization",
    author = "Sharma, Eva  and
      Li, Chen  and
      Wang, Lu",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1212",
    doi = "10.18653/v1/P19-1212",
    pages = "2204--2213",
    abstract = "Most existing text summarization datasets are compiled from the news domain, where summaries have a flattened discourse structure. In such datasets, summary-worthy content often appears in the beginning of input articles. Moreover, large segments from input articles are present verbatim in their respective summaries. These issues impede the learning and evaluation of systems that can understand an article{'}s global content structure as well as produce abstractive summaries with high compression ratio. In this work, we present a novel dataset, BIGPATENT, consisting of 1.3 million records of U.S. patent documents along with human written abstractive summaries. Compared to existing summarization datasets, BIGPATENT has the following properties: i) summaries contain a richer discourse structure with more recurring entities, ii) salient content is evenly distributed in the input, and iii) lesser and shorter extractive fragments are present in the summaries. Finally, we train and evaluate baselines and popular learning models on BIGPATENT to shed light on new challenges and motivate future directions for summarization research.",
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

class BigPatentConfig(datalabs.BuilderConfig):
    """BuilderConfig for BigPatent."""

    def __init__(self, **kwargs):
        """BuilderConfig for BigPatent.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BigPatentConfig, self).__init__(**kwargs)


class BigPatentDataset(datalabs.GeneratorBasedBuilder):
    """BigPatent Dataset."""
    _FILE_ID = "1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"
    BUILDER_CONFIGS = [
        BigPatentConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="BigPatent dataset for summarization, document",
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
            homepage="https://evasharma.github.io/bigpatent/",
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
        train_path = dl_manager.extract(os.path.join(f_path, "bigPatentData", "train.tar.gz"))
        test_path = dl_manager.extract(os.path.join(f_path, "bigPatentData", "test.tar.gz"))
        val_path = dl_manager.extract(os.path.join(f_path, "bigPatentData", "val.tar.gz"))  

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": os.path.join(train_path, "train")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": os.path.join(val_path, "val")}
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": os.path.join(test_path, "test")}
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate BigPatent examples."""
        data_dirs = os.listdir(f_path)
        data_dirs = [x for x in data_dirs if len(x) == 1]
        cnt = 0
        for data_dir in data_dirs:
            cur_dir = os.path.join(f_path, data_dir)
            file_names = os.listdir(cur_dir)
            for file_name in file_names:
                with gzip.open(os.path.join(cur_dir, file_name), 'r') as f:
                    for row in f:
                        data = json.loads(row)
                        yield cnt, {
                            _ARTICLE: data["description"],
                            _ABSTRACT: data["abstract"],
                        }
                        cnt += 1