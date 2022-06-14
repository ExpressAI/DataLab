import csv
import os

import datalabs
from datalabs import get_task, TaskType


_CITATION = """\
@article{DBLP:journals/corr/LowePSP15,
  author    = {Ryan Lowe and
               Nissan Pow and
               Iulian Serban and
               Joelle Pineau},
  title     = {The Ubuntu Dialogue Corpus: {A} Large Dataset for Research in Unstructured
               Multi-Turn Dialogue Systems},
  journal   = {CoRR},
  volume    = {abs/1506.08909},
  year      = {2015},
  url       = {http://arxiv.org/abs/1506.08909},
  archivePrefix = {arXiv},
  eprint    = {1506.08909},
  timestamp = {Mon, 13 Aug 2018 16:48:23 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LowePSP15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
Ubuntu Dialogue Corpus, a dataset containing almost 1 million multi-turn dialogues, with a total of over 7 million utterances and 100 million words. This provides a unique resource for research into building dialogue managers based on neural language models that can make use of large amounts of unlabeled data. The dataset has both the multi-turn property of conversations in the Dialog State Tracking Challenge datasets, and the unstructured nature of interactions from microblog services such as Twitter.
"""
_TRAIN_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/train.csv"
_VALIDATION_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/valid.csv"
_TEST_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/ubuntu_dialogs_corpus/test.csv"


class UbuntuDialogsCorpus(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
        
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                "context": datalabs.features.Sequence(datalabs.Value("string")),
                "utterance": datalabs.Value("string"),
                "label": datalabs.Value("int32") 
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/rkadlec/ubuntu-ranking-dataset-creator",
            citation=_CITATION,
            languages = ["en"],
            task_templates=[
                get_task(TaskType.retrieval_based_dialogue)(
                    context_column="context", utterance_column="utterance",label_column="label"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        

        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path,'split':'train'}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path,'split':'valid'}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path,'split':'test'})
        ]


    def _generate_examples(self, filepath,split):
        
        with open(filepath, encoding="utf-8") as f:
            data = csv.DictReader(f)
            count=-1
            for id_, row in enumerate(data):
                context=row["Context"].strip().split('__eou__ __eot__')[:-1]

                if split=='train':
                    yield id_, {
                        "context": context,
                        "utterance": row["Utterance"],
                        "label": row[ "Label"]
                    }
                else:
                    conversations=[{
                        "context": context,
                        "utterance": row["Ground Truth Utterance"],
                        "label": 1
                    }]

                    for i in range(9):
                        conversations.append({
                            "context": context,
                            "utterance": row["Distractor_" + str(i)],
                            "label": 0
                            }
                        )

                    for conversation in conversations:
                        count+=1
                        yield count, conversation