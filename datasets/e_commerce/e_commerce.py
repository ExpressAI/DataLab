import csv
import os

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Features, Value, Sequence


_CITATION = """\
@inproceedings{zhang2018dua,
    title = {Modeling Multi-turn Conversation with Deep Utterance Aggregation},
    author = {Zhang, Zhuosheng and Li, Jiangtong and Zhu, Pengfei and Zhao, Hai},
    booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
    pages={3740--3752},
    year = {2018}
}
"""

_DESCRIPTION = """\
E-commerce Dialogue Corpus, comprising a training data set, a development set and a test set for retrieval based chatbot.
"""
_TRAIN_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/E-commerce/train_revised.txt"
_VALIDATION_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/E-commerce/dev_revised.txt"
_TEST_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/E-commerce/test_revised.txt"


class E_commerce(datalabs.GeneratorBasedBuilder):
   
    VERSION = datalabs.Version("1.0.0")

    def _info(self):
        
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features(
                {
                "context": Sequence(Value("string")),
                "utterance": Sequence(Value("string")),
                "label": Value("int32") 
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/cooelf/DeepUtteranceAggregation",
            citation=_CITATION,
            languages = ["zh"],
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
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        
        with open(filepath, encoding="utf-8") as f:
            
            for id_, row in enumerate(f):
                line=row.strip().split('##')
                yield id_, {
                    "context": line[0].split('\t'),
                    "utterance": line[1].split('\t'),
                    "label":int(line[2])
                }