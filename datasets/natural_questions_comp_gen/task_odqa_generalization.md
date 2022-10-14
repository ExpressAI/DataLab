# Analyzing the Generalization of Open Domain Question Answering

In this file we describe how to analyze the generalization of models trained on open domain question answering datasets, for example [`natural questions`](https://github.com/google-research-datasets/natural-questions).

## Data Preparation

In order to perform analysis of your results, the predicted answers should be in the following
text format:

```
william henry bragg
may 18, 2018
...
```

where each line is an answer prediction.

An example system output file is here:

* [test.dpr.nq.txt](https://github.com/likicode/QA-generalize/blob/master/predictions/test.dpr.nq.txt)

## Performing Basic Analysis

The below example loads the `natural_questions_comp_gen` dataset from DataLab.

```shell
explainaboard --task qa_open_domain --dataset natural_questions_comp_gen --system_outputs MY_FILE > report.json
```

* `--task`: denotes the task name.
* `--system_outputs`: denote the path of system outputs. Multiple one should be
  separated by space, for example, system1 system2
* `--dataset`:optional, denotes the dataset name
* `report.json`: the generated analysis file with json format. . Tips: use a json viewer
  like [this one](http://jsonviewer.stack.hu/) for better interpretation.
