# Progress
List datasets which are supported by SDK and their associated information w.r.t task schema 


|Datasets|Updated Date|Task Schema|Normalized State|Comments|Constructor|
|:---    |:---        |:---       |:---       |:---    |:---    |
[govreport](https://github.com/ExpressAI/DataLab/blob/main/datasets/govreport/govreport.py)|2022-02-01|[Summarization](https://github.com/ExpressAI/DataLab/blob/451bab322a190cc1b1b3610c7d802bd0d0f33c00/src/datalabs/tasks/summarization.py#L22)|Done|Current definition: `text`,  `summary`|yixinliu|
[duorc](https://github.com/ExpressAI/DataLab/blob/main/datasets/duorc/duorc.py)|2022-02-03|[QuestionAnsweringExtractive](https://github.com/ExpressAI/DataLab/blob/adddf0071d0826e090b9ddecd7a98a09e8b625e4/src/datalabs/tasks/question_answering.py#L9)|Pending|different ids (`plot_id`, `q_id`) should be unified|jinlanfu|
[wiki_hop](https://github.com/ExpressAI/DataLab/blob/main/datasets/wiki_hop/wiki_hop.py)|2022-02-03|[QuestionAnsweringExtractive](https://github.com/ExpressAI/DataLab/blob/adddf0071d0826e090b9ddecd7a98a09e8b625e4/src/datalabs/tasks/question_answering.py#L9)|Pending|Two new fields: `candidates`, `annotations`|jinlanfu|
[hotpot_qa](https://github.com/ExpressAI/DataLab/blob/main/datasets/hotpot_qa/hotpot_qa.py)|2022-02-03|[QuestionAnsweringHotpot](https://github.com/ExpressAI/DataLab/blob/cf2471c750fbe33325b482bb5b1d9ea6fc734d56/src/datalabs/tasks/question_answering.py#L35)|Pending|Many new fileds (`supporting_facts`), `context` will be a json with a list of sentences.|jinlanfu|
[spider](https://github.com/ExpressAI/DataLab/blob/main/datasets/spider/spider.py)|2022-02-03|[SemanticParsing](https://github.com/ExpressAI/DataLab/blob/cf2471c750fbe33325b482bb5b1d9ea6fc734d56/src/datalabs/tasks/semantic_parsing.py#L9)|Pending|Current definition: `question`,  `query`|jinlanfu|
[atis](https://github.com/ExpressAI/DataLab/blob/main/datasets/atis/atis.py)|2022-02-05|[TextClassification](https://github.com/ExpressAI/DataLab/blob/a291d0a94e01b1948f915afc354a7e207ca1a906/src/datalabs/tasks/text_classification.py#L22)|Done|Current definition:`text`,`label`|weizhe|
[cr](https://github.com/ExpressAI/DataLab/blob/main/datasets/cr/cr.py)|2022-02-06|[TextClassification](https://github.com/ExpressAI/DataLab/blob/a291d0a94e01b1948f915afc354a7e207ca1a906/src/datalabs/tasks/text_classification.py#L22)|Done|Current definition:`text`,`label`|weizhe|
[mr](https://github.com/ExpressAI/DataLab/blob/main/datasets/mr/mr.py)|2022-02-06|[TextClassification](https://github.com/ExpressAI/DataLab/blob/a291d0a94e01b1948f915afc354a7e207ca1a906/src/datalabs/tasks/text_classification.py#L22)|Done|Current definition:`text`,`label`|weizhe|
[qc](https://github.com/ExpressAI/DataLab/blob/main/datasets/qc/qc.py)|2022-02-06|[TextClassification](https://github.com/ExpressAI/DataLab/blob/a291d0a94e01b1948f915afc354a7e207ca1a906/src/datalabs/tasks/text_classification.py#L22)|Done|Current definition:`text`,`label`|weizhe|
[subj](https://github.com/ExpressAI/DataLab/blob/main/datasets/subj/subj.py)|2022-02-06|[TextClassification](https://github.com/ExpressAI/DataLab/blob/a291d0a94e01b1948f915afc354a7e207ca1a906/src/datalabs/tasks/text_classification.py#L22)|Done|Current definition:`text`,`label`|weizhe|
[afqmc](https://github.com/ExpressAI/DataLab/blob/main/datasets/afqmc/afqmc.py)|2022-02-06|[TextMatching](https://github.com/ExpressAI/DataLab/blob/c20c79a2276e87ac13a541878a6ae47efddcad31/src/datalabs/tasks/text_matching.py#L22)|Done|Current definition:`text1`,`text2`, `label`|zhengfu|


