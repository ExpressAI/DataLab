import csv
import os

import datalabs
from datalabs import get_task, TaskType


_CITATION = """\
@article{wu2016sequential,
  title={Sequential matching network: A new architecture for multi-turn response selection in retrieval-based chatbots},
  author={Wu, Yu and Wu, Wei and Xing, Chen and Zhou, Ming and Li, Zhoujun},
  journal={arXiv preprint arXiv:1612.01627},
  year={2016}
}
"""

_DESCRIPTION = """\
The Douban Conversation Corpus is a data set with open domain conversations. Response candidates in the test set of the Douban Conversation.
Corpus are collected following the procedure of a retrieval-based chatbot and are labeled by human judges. It simulates the real scenario of a retrievalbased chatbot
"""
_TRAIN_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/DoubanConversaionCorpus/train.txt"
_VALIDATION_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/DoubanConversaionCorpus/dev.txt"
_TEST_DOWNLOAD_URL="https://cdatalab1.oss-cn-beijing.aliyuncs.com/dialogue/DoubanConversaionCorpus/test.txt"


class DoubanConversaion(datalabs.GeneratorBasedBuilder):
   
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
            homepage="https://github.com/MarkWuNLP/MultiTurnResponseSelection",
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
                line=row.strip().split('\t')
                yield id_, {
                    "context": line[1:-1],
                    "utterance": line[-1],
                    "label":int(line[0])
                }