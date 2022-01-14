
import json
import os
import datalabs
from datalabs.tasks import SequenceLabeling
from datalabs.task_dataset import SequenceLabelingDataset
logger = datalabs.logging.get_logger(__name__)

_CITATION = """\
    @data{AB2/MKJJ2R_2013,
    author = {Weischedel, Ralph and Palmer, Martha and Marcus, Mitchell and Hovy, Eduard and Pradhan, Sameer and Ramshaw, Lance and Xue, Nianwen and Taylor, Ann and Kaufman, Jeff and Franchini, Michelle and El-Bachouti, Mohammed and Belvin, Robert and Houston, Ann},
    publisher = {Abacus Data Network},
    title = {{OntoNotes Release 5.0}},
    year = {2013},
    version = {V1},
    doi = {11272.1/AB2/MKJJ2R},
    url = {https://hdl.handle.net/11272.1/AB2/MKJJ2R}
    }
"""
_DESCRIPTION = """\
OntoNotes Release 5.0 is the final release of the OntoNotes project, a collaborative effort between
 BBN Technologies, the University of Colorado, the University of Pennsylvania and the University of Southern 
 Californias Information Sciences Institute. The goal of the project was to annotate a large corpus comprising 
 various genres of text (news, conversational telephone speech, weblogs, usenet newsgroups, broadcast, talk shows)
  in three languages (English, Chinese, and Arabic) with structural
 information (syntax and predicate argument structure) and shallow semantics (word sense linked to an ontology and coreference). 
"""
_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC2013T19"
_LICENSE = "LDC User Agreement for Non-Members"
_URL = "https://raw.githubusercontent.com/expressai/data/master/rawdata/ner_on.zip"


config_maps = {
    "notebc":{
        "labels":['O', 'B-GPE', 'B-PERSON', 'I-PERSON', 'B-FAC', 'I-FAC', 'B-ORG', 'B-NORP', 'I-ORG', 'B-LAW', 'I-LAW',
                  'B-LOC', 'I-LOC', 'B-CARDINAL', 'I-GPE', 'B-QUANTITY', 'I-QUANTITY', 'B-DATE', 'I-DATE', 'B-LANGUAGE',
                  'B-EVENT', 'I-EVENT', 'B-PERCENT', 'I-PERCENT', 'B-ORDINAL', 'B-TIME', 'I-TIME', 'B-MONEY',
                  'I-MONEY', 'I-CARDINAL', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-PRODUCT', 'I-NORP', 'I-PRODUCT', 'I-LANGUAGE', 'I-ORDINAL'],
    },
    "notebn":{
        "labels":['O', 'B-GPE', 'B-PERSON', 'I-PERSON', 'B-FAC', 'I-FAC', 'B-ORG', 'B-NORP', 'I-ORG',
                  'B-LAW', 'I-LAW', 'B-LOC', 'I-LOC', 'B-CARDINAL', 'I-GPE', 'B-QUANTITY', 'I-QUANTITY',
                  'B-DATE', 'I-DATE', 'B-LANGUAGE', 'B-EVENT', 'I-EVENT', 'B-PERCENT', 'I-PERCENT', 'B-ORDINAL',
                  'B-TIME', 'I-TIME', 'B-MONEY', 'I-MONEY', 'I-CARDINAL',
                  'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-PRODUCT', 'I-NORP', 'I-PRODUCT', 'I-LANGUAGE', 'I-ORDINAL'],
    },
    "notemz":{
        "labels": ['O', 'B-NORP', 'I-NORP', 'B-DATE', 'I-DATE', 'B-CARDINAL', 'I-CARDINAL', 'B-GPE', 'B-FAC',
                   'I-FAC', 'B-PERCENT', 'I-PERCENT', 'B-ORG', 'I-ORG', 'I-GPE', 'B-TIME', 'I-TIME', 'B-PERSON',
                   'I-PERSON', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-ORDINAL', 'B-LANGUAGE', 'B-QUANTITY', 'I-QUANTITY', 'B-LOC', 'I-LOC',
                   'B-EVENT', 'I-LANGUAGE', 'I-EVENT', 'B-MONEY', 'I-MONEY', 'B-PRODUCT', 'B-LAW', 'I-LAW', 'I-PRODUCT', 'I-ORDINAL'],
    },
    "notenw":{
        "labels": ['B-ORG', 'I-ORG', 'O', 'B-QUANTITY', 'I-QUANTITY', 'B-PERSON', 'I-PERSON', 'B-GPE', 'I-GPE', 'B-MONEY',
                   'I-MONEY', 'B-DATE', 'I-DATE', 'B-EVENT', 'I-EVENT', 'B-NORP', 'B-LANGUAGE', 'B-ORDINAL', 'B-FAC', 'I-FAC',
                   'B-TIME', 'I-TIME', 'B-LOC', 'I-LOC', 'B-CARDINAL', 'B-PERCENT', 'I-PERCENT', 'I-NORP', 'B-WORK_OF_ART',
                   'I-WORK_OF_ART', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-PRODUCT', 'I-PRODUCT', 'I-ORDINAL', 'I-LANGUAGE'],
    },
    "notetc":{
        "labels": ['O', 'B-PERSON', 'I-PERSON', 'B-CARDINAL', 'I-CARDINAL', 'B-ORG', 'B-GPE', 'B-FAC', 'I-FAC',
                   'B-MONEY', 'I-MONEY', 'I-ORG', 'B-QUANTITY', 'I-QUANTITY', 'B-LOC', 'B-NORP', 'B-DATE',
                   'I-DATE', 'I-GPE', 'B-TIME', 'B-ORDINAL', 'B-PRODUCT', 'I-TIME', 'I-NORP', 'B-PERCENT', 'I-PRODUCT', 'B-LANGUAGE', 'I-LANGUAGE',
                   'I-LOC', 'I-PERCENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-EVENT', 'I-EVENT'],
    },
    "notewb":{
        "labels": ['O', 'B-CARDINAL', 'B-GPE', 'B-DATE', 'I-DATE', 'B-QUANTITY', 'I-QUANTITY', 'B-PERSON', 'I-PERSON',
                   'B-ORG', 'I-ORG', 'B-LOC', 'B-PRODUCT', 'I-CARDINAL', 'I-LOC', 'B-NORP', 'I-NORP', 'B-PERCENT',
                   'I-PERCENT', 'B-MONEY', 'I-MONEY', 'I-GPE', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-EVENT', 'I-EVENT', 'B-LAW',
                   'I-LAW', 'B-TIME', 'I-TIME', 'B-FAC', 'I-FAC', 'B-ORDINAL', 'I-PRODUCT', 'B-LANGUAGE', 'I-ORDINAL'],
    },
}



class OntonotesNERConfig(datalabs.BuilderConfig):

    def __init__(self,
                 labels = None,
                 **kwargs):
        super(OntonotesNERConfig, self).__init__(**kwargs)
        self.labels = labels


class OntonotesNER(datalabs.GeneratorBasedBuilder):

    def __init__(self,*args, **kwargs):
        super(OntonotesNER, self).__init__(*args, **kwargs)
        self.dataset_class = SequenceLabelingDataset


    VERSION = datalabs.Version("1.0.0")

    BUILDER_CONFIGS = [
        OntonotesNERConfig(
            name="{}".format(domain),
            version=datalabs.Version("1.0.0"),
            labels = val["labels"]
        )
        for domain, val in config_maps.items()
    ]

    # DEFAULT_CONFIG_NAME = "notebc"


    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                    "id": datalabs.Value("string"),
                    "tokens": datalabs.Sequence(datalabs.Value("string")),
                    "tags": datalabs.Sequence(
                        datalabs.features.ClassLabel(
                            names=self.config.labels
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            version=self.VERSION,
            task_templates=[SequenceLabeling(tokens_column="tokens", tags_column="tags")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        domain = str(self.config.name)
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/ner_on/" + domain, "train-" + domain + ".tsv"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/ner_on/" + domain, "test-" + domain + ".tsv"),
                },
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "/ner_on/" + domain, "dev-" + domain + ".tsv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    # New sentence
                    if not current_tokens:
                        # Consecutive empty lines will cause empty sentences
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    sentence = (
                        sentence_counter,
                        {
                            "id": str(sentence_counter),
                            "tokens": current_tokens,
                            "tags": current_labels,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                yield sentence_counter, {
                    "id": str(sentence_counter),
                    "tokens": current_tokens,
                    "tags": current_labels,
                }
