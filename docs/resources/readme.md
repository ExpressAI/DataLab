# Introduction of Resources


## Operation Information

All information is stored in the file `operations_info.json`
where:

(1) `class_type` represents the operation type
* aggregating: aggregation operation
* editing: edit operation
* preprocessing: preprocess operation
* prompting: prompt operation
* featurizing: featurize operation

(2) `args` describe the operation from diverse aspects:
* name: function name
* contributor: the original contributor of the function
* task: tasks that the function can support
    - it is a string with one or multiple task names or task categories
    - `Any` suggests that this function can be applied to any tasks
* description: the descriptive information of the function

