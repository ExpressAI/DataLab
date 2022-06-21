# How to Create an Interactive Benchmark？

 
Just two step!!


## 1. Add your Datasets
To create an interactive benchmark, a necessary premise is that all datasets in your 
benchmark should be covered by [DataLab](). So,
* (1) you first need to check whether all your datasets have been already supported.
If yes, directly conduct the next step. Otherwise, you need to
* (2) add your datasets into [DataLab]() following this [documentation](). 




## 2. Write a Benchmark Config File
Benchmark config defines how benchmark is composed from diverse datasets. Specifically,
it is a JSON file with following format:

```JSON
{
  "id": "miniglue",
  "name": "miniGLUE",
  "description":"miniGLUE",
  "logo":"https://explainaboard.s3.amazonaws.com/benchmarks/figures/gaokao.jpg",
  "datasets": [
    {"dataset_name": "sst2", "task": "text-classification", "metrics": [{"name": "F1", "weight": 0.2, "default": 0.0},
                                                                        {"name": "Accuracy", "weight": 0.8, "default":  0.0}]},
    {"dataset_name": "snli", "task": "natural-language-inference", "metrics": [{"name": "Accuracy"}]}
  ],
  "views": [
    {
      "name": "Overall",
      "operations": [
        {"op": "weighted_sum", "group_by": "dataset_name", "weight": "metric_weight"},
        {"op": "mean"}
      ]
    },
    {
      "name": "Task-weighted Overall",
      "operations": [
        {"op": "weighted_sum", "group_by": ["dataset_name", "task"], "weight": "metric_weight"},
        {"op": "weighted_sum", "weight": "task", "weight_map": {"text-classification":  0.7, "natural-language-inference": 0.3}}
      ]
    },
    {
      "name": "Task-wise Average",
      "operations": [
        {"op": "weighted_sum", "group_by": ["dataset_name", "task"], "weight": "metric_weight"},
        {"op": "mean", "group_by": "task"}
      ]
    }
  ],
  "default_views": ["Overall"]
}
```