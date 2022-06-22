"""ScisummNet: A Large Annotated Corpus and Content-Impact Models for Scientific Paper Summarization with Citation Networks."""
import os
import json
import glob
from xml.dom import minidom
import datalabs
from datalabs import get_task, TaskType
from datalabs.tasks.summarization import _MDS_TEXT_COLUMN

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
)
from datalabs.utils.more_features import (
    get_feature_schemas,
)

_CITATION = """\
@inproceedings{yasunaga2019scisummnet,
  title={Scisummnet: A large annotated corpus and content-impact models for scientific paper summarization with citation networks},
  author={Yasunaga, Michihiro and Kasai, Jungo and Zhang, Rui and Fabbri, Alexander R and Li, Irene and Friedman, Dan and Radev, Dragomir R},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={7386--7393},
  year={2019}
}
"""

_DESCRIPTION = """\
We developed the first large-scale, human-annotated Scisumm dataset, ScisummNet. 
It provides over 1,000 papers in the ACL anthology network with their citation networks 
(e.g. citation sentences, citation counts) and their comprehensive, manual summaries.
see: https://arxiv.org/pdf/1909.01716.pdf
"""

_HOMEPAGE = "https://cs.stanford.edu/~myasu/projects/scisumm_net/"
_LICENSE = "CC BY-SA 4.0"
_ABSTRACT = "summary"
_ARTICLE = "text"


class ScisummNetConfig(datalabs.BuilderConfig):
    """BuilderConfig for ScisummNet."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for ScisummNet.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ScisummNetConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class ScisummNetDataset(datalabs.GeneratorBasedBuilder):
    """ScisummNet Dataset."""

    _FILE_ID = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"

    BUILDER_CONFIGS = [
        ScisummNetConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="ScisummNet dataset for scientific paper summarization, single document version.",
            task_templates=[get_task(TaskType.summarization)(
                source_column=_ARTICLE,
                reference_column=_ABSTRACT)]
        ),
        ScisummNetConfig(
            name="multidoc",
            version=datalabs.Version("1.0.0"),
            description="ScisummNet dataset for scientific paper summarization, multi-document document version.",
            task_templates=[get_task(TaskType.multi_doc_summarization)(
                source_column=_MDS_TEXT_COLUMN,
                reference_column=_ABSTRACT)]
        )
    ]
    DEFAULT_CONFIG_NAME = "multidoc"

    def _info(self):
        # Should return a datalab.DatasetInfo object
        features_dataset = {}

        if self.config.name == "document":
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                        get_schema_of_sample_level_features)
        elif self.config.name == "multidoc":
            features_sample = datalabs.Features(
                {
                    _MDS_TEXT_COLUMN: datalabs.Sequence(datalabs.Value("string")),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )

        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            languages=["en"],
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):

        f_path = dl_manager.download_and_extract(self._FILE_ID)

        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={"f_path": os.path.join(f_path, "scisummnet_release1.1__20190413")}
            )
        ]

    def _generate_examples(self, f_path):
        # """Generate ScisummNet examples."""
        paper_dir = os.path.join(f_path, "top1000_complete")
        paper_ids = os.listdir(paper_dir)
        paper_ids.sort()
        paper_paths = [os.path.join(paper_dir, paper_id) for paper_id in paper_ids if "." not in paper_id]
        datas = []
        for paper_path in paper_paths:
            # process document
            document_xml = glob.glob(os.path.join(paper_path, "Documents_xml", "*.xml"))[0]
            document_file = minidom.parse(document_xml)
            root = document_file.documentElement
            try:
                abstract_elem = root.getElementsByTagName('ABSTRACT')[0]
                abstract_sentences = abstract_elem.getElementsByTagName('S')
                abstract = []
                for abstract_sentence in abstract_sentences:
                    abstract.append(abstract_sentence.firstChild.data)
                abstract = " ".join(abstract)
            except:
                # TODO some papers do not contain abstract section.
                abstract = ""
                continue

            # process summary
            summary_txt = glob.glob(os.path.join(paper_path, "summary", "*.txt"))[0]
            summary_f = open(summary_txt, encoding="utf-8")
            lines = summary_f.readlines()
            summary = " ".join([line.strip() for line in lines])

            # process citing sentences
            f = open(os.path.join(paper_path, "citing_sentences_annotated.json"), encoding="utf-8")
            citing_sentences = json.load(f)
            clean_texts = []
            for citing_sentence in citing_sentences:
                clean_text = citing_sentence["clean_text"]
                if clean_text.strip():
                    clean_texts.append(clean_text)

            texts = []
            if abstract:
                texts.append(abstract)
            texts.extend(clean_texts)
            datas.append((texts, summary))

        if "document" in self.config.name:
            for id_, (texts, summary) in enumerate(datas):
                raw_feature_info = {
                    _ARTICLE: " ".join(texts),
                    _ABSTRACT: summary
                }
                if not self.feature_expanding:
                    yield id_, raw_feature_info
                else:
                    additional_feature_info = get_features_sample_level(raw_feature_info)
                    raw_feature_info.update(additional_feature_info)
                    yield id_, raw_feature_info
        elif "multidoc" in self.config.name:
            for id_, (texts, summary) in enumerate(datas):
                raw_feature_info = {
                    _MDS_TEXT_COLUMN: texts,
                    _ABSTRACT: summary
                }
                yield id_, raw_feature_info
