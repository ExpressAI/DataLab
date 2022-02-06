# Progress
List datasets which are supported by SDK and their associated information w.r.t task schema 


|Datasets|Updated Date|Task Schema|Normalized State|Comments|Constructor|
|:---    |:---        |:---       |:---       |:---    |:---    |
[duorc](https://github.com/ExpressAI/DataLab/blob/main/datasets/duorc/duorc.py)|2022-02-03|[QuestionAnsweringExtractive](https://github.com/ExpressAI/DataLab/blob/adddf0071d0826e090b9ddecd7a98a09e8b625e4/src/datalabs/tasks/question_answering.py#L9)|Pending|different ids (`plot_id`, `q_id`) should be unified|jinlanfu|
[wiki_hop](https://github.com/ExpressAI/DataLab/blob/main/datasets/wiki_hop/wiki_hop.py)|2022-02-03|[QuestionAnsweringExtractive](https://github.com/ExpressAI/DataLab/blob/adddf0071d0826e090b9ddecd7a98a09e8b625e4/src/datalabs/tasks/question_answering.py#L9)|Pending|Two new fields: `candidates`, `annotations`|jinlanfu|
[hotpot_qa](https://github.com/ExpressAI/DataLab/blob/main/datasets/hotpot_qa/hotpot_qa.py)|2022-02-03|[QuestionAnsweringHotpot](https://github.com/ExpressAI/DataLab/blob/cf2471c750fbe33325b482bb5b1d9ea6fc734d56/src/datalabs/tasks/question_answering.py#L35)|Pending|Many new fileds (`supporting_facts`), `context` will be a json with a list of sentences.|jinlanfu|
[Spider](https://github.com/ExpressAI/DataLab/blob/main/datasets/spider/spider.py)|2022-02-03|[SemanticParsing](https://github.com/ExpressAI/DataLab/blob/cf2471c750fbe33325b482bb5b1d9ea6fc734d56/src/datalabs/tasks/semantic_parsing.py#L9)|Pending|Current definition: `question`,  `query`|jinlanfu|


