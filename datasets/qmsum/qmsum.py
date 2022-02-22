# following the data process of https://github.com/Yale-LILY/QMSum/blob/main/data_process.ipynb

import json
import os
import datalabs
from datalabs.tasks import Summarization, QuerySummarization
from nltk import word_tokenize

# the following package are needed when more additional features are expected to be calculated
from featurize.summarization import (
    get_features_sample_level,
    get_schema_of_sample_level_features,
    )
from datalabs.utils.more_features import (
    get_feature_schemas,
)



_DESCRIPTION = """
 QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task, which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
 See: https://aclanthology.org/2021.naacl-main.472.pdf
 See: https://github.com/Yale-LILY/QMSum
"""
_CITATION = """\
    @inproceedings{zhong-etal-2021-qmsum,
    title = "{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization",
    author = "Zhong, Ming  and
      Yin, Da  and
      Yu, Tao  and
      Zaidi, Ahmad  and
      Mutuma, Mutethia  and
      Jha, Rahul  and
      Awadallah, Ahmed Hassan  and
      Celikyilmaz, Asli  and
      Liu, Yang  and
      Qiu, Xipeng  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.472",
    doi = "10.18653/v1/2021.naacl-main.472",
    pages = "5905--5921",
    abstract = "Meetings are a key component of human collaboration. As increasing numbers of meetings are recorded and transcribed, meeting summaries have become essential to remind those who may or may not have attended the meetings about the key decisions made and the tasks to be completed. However, it is hard to create a single short summary that covers all the content of a long meeting involving multiple people and topics. In order to satisfy the needs of different types of users, we define a new query-based multi-domain meeting summarization task, where models have to select and summarize relevant spans of meetings in response to a query, and we introduce QMSum, a new benchmark for this task. QMSum consists of 1,808 query-summary pairs over 232 meetings in multiple domains. Besides, we investigate a locate-then-summarize method and evaluate a set of strong summarization baselines on the task. Experimental results and manual analysis reveal that QMSum presents significant challenges in long meeting summarization for future research.",
}
"""
_ABSTRACT = "summary"
_ARTICLE = "text"
_KEY = "query"

# tokneize a sent
def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens

# filter some noises caused by speech recognition
def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text

class QMSumConfig(datalabs.BuilderConfig):
    """BuilderConfig for QMSum."""

    def __init__(self, task_templates, **kwargs):
        """BuilderConfig for QMSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QMSumConfig, self).__init__(**kwargs)
        self.task_templates = task_templates


class QMSumDataset(datalabs.GeneratorBasedBuilder):
    """QMSum Dataset."""
    _TRAIN_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/train.jsonl"
    _VAL_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/val.jsonl"
    _TEST_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/test.jsonl"
    BUILDER_CONFIGS = [
        QMSumConfig(
            name="document",
            version=datalabs.Version("1.0.0"),
            description="QMSum dataset for summarization, single document summarization version",
            task_templates=[Summarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT)]
        ),
        QMSumConfig(
            name="query-based",
            version=datalabs.Version("1.0.0"),
            description="QMSum dataset for summarization, query-based summarization version",
            task_templates=[QuerySummarization(
                text_column=_ARTICLE, summary_column=_ABSTRACT, query_column=_KEY)]
        ),
    ]
    DEFAULT_CONFIG_NAME = "query-based"

    def _info(self):
        features_dataset = {}
        # Should return a datalab.DatasetInfo object
        if "document" in self.config.name:

            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                }
            )
            if self.feature_expanding:
                features_sample, features_dataset = get_feature_schemas(features_sample,
                                                                        get_schema_of_sample_level_features)

        else:
            features_sample = datalabs.Features(
                {
                    _ARTICLE: datalabs.Value("string"),
                    _ABSTRACT: datalabs.Value("string"),
                    _KEY: datalabs.Value("string"),
                }
            )
        return datalabs.DatasetInfo(
            description=_DESCRIPTION,
            features=features_sample,
            features_dataset=features_dataset,
            supervised_keys=None,
            homepage="https://github.com/Yale-LILY/QMSum",
            citation=_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download(self._TRAIN_URL)
        val_path = dl_manager.download(self._VAL_URL)
        test_path = dl_manager.download(self._TEST_URL)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN, gen_kwargs={"f_path": train_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.VALIDATION, gen_kwargs={"f_path": val_path},
            ),
            datalabs.SplitGenerator(
                name=datalabs.Split.TEST, gen_kwargs={"f_path": test_path},
            ),
        ]

    def _generate_examples(self, f_path):
        """Generate QMSum examples."""
        _id = 0
        with open(f_path) as f:
            for x in f:
                data = json.loads(x)
                src = []
                for k in range(len(data['meeting_transcripts'])):
                    cur_turn = data['meeting_transcripts'][k]['speaker'].lower() + ': '
                    cur_turn = cur_turn + tokenize(data['meeting_transcripts'][k]['content'])
                    src.append(cur_turn)
                src = ' '.join(src)
                for j in range(len(data['general_query_list'])):
                    cur = {}
                    query = tokenize(data['general_query_list'][j]['query'])
                    if "document" in self.config.name:
                        cur['text'] = clean_data('<s> ' + query + ' </s> ' + src + ' </s>')
                        target = tokenize(data['general_query_list'][j]['answer'])
                        cur['summary'] = target

                        raw_feature_info = cur

                        if not self.feature_expanding:
                            yield id_, raw_feature_info
                        else:
                            additional_feature_info = get_features_sample_level(raw_feature_info)
                            raw_feature_info.update(additional_feature_info)
                            # print(additional_feature_info)
                            yield id_, raw_feature_info




                    else:
                        query, src = clean_data(query), clean_data(src)
                        target = tokenize(data['general_query_list'][j]['answer'])
                        yield _id, {_ARTICLE: src, _ABSTRACT: target, _KEY: query}
                    _id += 1   
                for j in range(len(data['specific_query_list'])):
                    cur = {}
                    query = tokenize(data['specific_query_list'][j]['query'])
                    if "document" in self.config.name:
                        cur['text'] = clean_data('<s> ' + query + ' </s> ' + src + ' </s>')
                        target = tokenize(data['specific_query_list'][j]['answer'])
                        cur['summary'] = target

                        raw_feature_info = cur

                        if not self.feature_expanding:
                            yield id_, raw_feature_info
                        else:
                            additional_feature_info = get_features_sample_level(raw_feature_info)
                            raw_feature_info.update(additional_feature_info)
                            # print(additional_feature_info)
                            yield id_, raw_feature_info



                    else:
                        query, src = clean_data(query), clean_data(src)
                        target = tokenize(data['specific_query_list'][j]['answer'])
                        yield _id, {_ARTICLE: src, _ABSTRACT: target, _KEY: query}
                    _id += 1
