# How to add a new task schema for your dataset?

### Background:
This will happen when you aim to add a new dataset whose task schema has NOT been supported by existing tasks schemas. In this case, you need to mannually add a new task schema.
Here are existing supported [task schema](https://github.com/ExpressAI/DataLab/tree/main/src/datalabs/tasks)


### Example

Suppose that we want to add the `sequence-labeling` as a new task schema, which requires three steps:

##### 1. creat a script for the class
We need to creat a script (`sequence_labeling.py`) in the [folder]((https://github.com/ExpressAI/DataLab/tree/main/src/datalabs/tasks) to claim the class `SequenceLabeling`(https://github.com/ExpressAI/DataLab/blob/main/src/datalabs/tasks/sequence_labeling.py).

##### 2. claim the new class in `__init__.py`
We then need to register the information of new class at [`__init__.py`](https://github.com/ExpressAI/DataLab/blob/main/src/datalabs/tasks/__init__.py)


