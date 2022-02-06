# How to add a new task schema for your dataset?

### Background:
This will happen when you aim to add a new dataset whose task schema has NOT been supported by existing tasks schemas. In this case, you need to mannually add a new task schema.
Here are existing supported [task schema](https://github.com/ExpressAI/DataLab/tree/main/src/datalabs/tasks)


### Example

Suppose that we want to add the `sequence-labeling` as a new task schema, which requires three steps:

##### 1. creat a script for the class
We need to creat a script (`sequence_labeling.py`) in the [folder](https://github.com/ExpressAI/DataLab/tree/main/src/datalabs/tasks) to claim the class [`SequenceLabeling`](https://github.com/ExpressAI/DataLab/blob/main/src/datalabs/tasks/sequence_labeling.py).

##### 2. claim the new class in `__init__.py`
We then need to register the information of new class at [`__init__.py`](https://github.com/ExpressAI/DataLab/blob/main/src/datalabs/tasks/__init__.py)


### Tips

* The motivation of introducing task schema is to help us easily standardize (normalize) different datasets from the same task category.
For example, the samples from both [`ag_news`](https://github.com/ExpressAI/DataLab/blob/main/datasets/ag_news/ag_news.py) (topic classification) and `sst2` (sentiment classification) should be [formatted as `text` and `label`](https://github.com/ExpressAI/DataLab/blob/da463705e983b771131c74ee5cef222d6d59d56e/src/datalabs/tasks/text_classification.py#L29). The advantage of doing this is we can easily process all datasets within this task category in a unified way (without any additional preprocessing).
* Once we introduce a new task schema, we can first refer to the schema of similar tasks and incrementally extend it. (`incrementally` kinda means partially `inherit` the similar task schema.)
For example, 
* you can refer to [QuestionAnsweringExtractive](https://github.com/ExpressAI/DataLab/blob/604656cdce05d539e94949f0c842fbbb5b368188/src/datalabs/tasks/question_answering.py#L9) if you aim to introduce other QA-based tasks.
* you can refer to [Summarization](https://github.com/ExpressAI/DataLab/blob/604656cdce05d539e94949f0c842fbbb5b368188/src/datalabs/tasks/summarization.py#L22) if other new generation tasks are being added.


