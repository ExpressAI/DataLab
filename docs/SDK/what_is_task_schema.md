# What is the `task schema`?

We introduce the concept of `task schema` that defines the format of datasets belonging to this task. This is useful to
standardize the formats of datasets from the same task.
For example, the schema of `text classification` task is:

* `text`:str
* `label`:ClassLabel

while for `text-matching` task, its schema can be defined as:
  
* `text1`:str
* `text2`:str
* `label`:ClassLabel

More detailed can refer to this [folder](https://github.com/ExpressAI/DataLab/tree/main/src/datalabs/tasks).

Also, this doc details [how to add a new task schema](add_new_task_schema.md).
