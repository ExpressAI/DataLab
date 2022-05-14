# coding=utf-8
# Copyright 2022 DataLab Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import datalabs
from datalabs import get_task, TaskType

_CITATION = '''\
@inproceedings{chen-etal-2018-bq,
    title = "The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification",
    author = "Chen, Jing  and
      Chen, Qingcai  and
      Liu, Xin  and
      Yang, Haijun  and
      Lu, Daohe  and
      Tang, Buzhou",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1536",
    doi = "10.18653/v1/D18-1536",
    pages = "4946--4951",
    abstract = "This paper introduces the Bank Question (BQ) corpus, a Chinese corpus for sentence semantic equivalence identification (SSEI). The BQ corpus contains 120,000 question pairs from 1-year online bank custom service logs. To efficiently process and annotate questions from such a large scale of logs, this paper proposes a clustering based annotation method to achieve questions with the same intent. First, the deduplicated questions with the same answer are clustered into stacks by the Word Mover{'}s Distance (WMD) based Affinity Propagation (AP) algorithm. Then, the annotators are asked to assign the clustered questions into different intent categories. Finally, the positive and negative question pairs for SSEI are selected in the same intent category and between different intent categories respectively. We also present six SSEI benchmark performance on our corpus, including state-of-the-art algorithms. As the largest manually annotated public Chinese SSEI corpus in the bank domain, the BQ corpus is not only useful for Chinese question semantic matching research, but also a significant resource for cross-lingual and cross-domain SSEI research. The corpus is available in public.",
}

'''

_DESCRIPTION = '''\
这个数据集可用于银行金融领域的问题匹配任务，包括了从线上银行系统日志里抽取的问题pair对，是目前最大的银行领域问题匹配数据集。
This dataset can be used for Text Matching task in the banking and finance field. 
It includes problem pairs extracted from online banking system logs 
and is the largest Text Matching dataset in the banking field.
For more information, please refer to https://aclanthology.org/D18-1536.
'''

_LICENSE = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bq_corpus/License.pdf"

# If you want to use the corpus, please download the application form, fill sign it respectively, 
# then fax or e-mail the scan version to Qingcai Chen (email: qingcai.chen@hit.edu.cn; Fax: +86-755-26033182.)
_UserAgreement = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bq_corpus/User_Agreement.pdf"

_TRAIN_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bq_corpus/train.tsv"
_VALIDATION_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bq_corpus/dev.tsv"
# _TEST_DOWNLOAD_URL = "https://cdatalab1.oss-cn-beijing.aliyuncs.com/text_matching/bq_corpus/test.tsv"


class BQCORPUS(datalabs.GeneratorBasedBuilder):
    def _info(self):
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=datalabs.Features({
                'text1': datalabs.Value('string'),
                'text2': datalabs.Value('string'),
                'label': datalabs.features.ClassLabel(names=['0', '1'])
            }),
            supervised_keys=None,
            homepage='https://aclanthology.org/D18-1536',
            citation=_CITATION,
            languages=["zh"],
            task_templates=[get_task(TaskType.paraphrase_identification)(
                text1_column="text1",
                text2_column="text2",
                label_column="label"),
            ],
        )

    def _split_generators(self, dl_manager):
        
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        # test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datalabs.SplitGenerator(name=datalabs.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datalabs.SplitGenerator(name=datalabs.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            # datalabs.SplitGenerator(name=datalabs.Split.TEST, gen_kwargs={"filepath": test_path})
        ]


    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = '\t')
            for id_, row in enumerate(csv_reader):
                if len(row) == 3:
                    text1, text2, label = row
                    label = int(label)
                    yield id_, {'text1': text1, 'text2': text2, 'label': label}
