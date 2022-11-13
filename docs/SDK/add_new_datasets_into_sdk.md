# How to add new datasets?

We will walk through how to add a new dataset into datalab.


## 1. Making your raw dataset available online
If your dataset already has a public link online, you can use that link.

Otherwise, you'll need to put your dataset into a server with downloadable links
(please make sure you have permission to redistribute the dataset first).
For example, you can place your datasets in
* google drive
* google cloud
* AWS S3
 
 
 
## 2. Create a new folder and write a data loader script.

Suppose the dataset name to be added is `cr`, we need to:
* create a folder `cr` in [DataLab/datasets/](https://github.com/ExpressAI/DataLab/tree/main/datasets)
* create a data loader script `cr.py` in the above folder, i.e., `Datalab/datasets/cr/cr.py`
* finish the data loader script based on some provided examples:
    * text-classification: [template](https://github.com/ExpressAI/DataLab/tree/main/datasets/cr)
    * extractive-qa: [template](https://github.com/ExpressAI/DataLab/blob/main/datasets/squad/squad.py)
    


## 3. Test in your local server
* enter into `Datalab/datasets` folder
* run following python commands

```python
   from datalabs import load_dataset
   dataset = load_dataset("./cr")
   print(dataset['train']._info)
   print(dataset['train']._info.task_templates)
```

## 4. Set up a pull request 
Once you successfully finished the above steps, if you would like to make your dataset
public, you can set up a pull request.

## 5. Make your datasets registered
Once you successfully added a new dataset, please update the the file [dataset_info_dev.jsonl](https://github.com/ExpressAI/DataLab/blob/main/utils/dataset_info_dev.jsonl)
by conducting the following command: 
```shell
python get_dataset_info.py --previous_jsonl dataset_info.jsonl --output_jsonl dataset_info_dev.jsonl --datasets YOUR_DATASET_NAME
cat dataset_info_dev.jsonl >> dataset_info.jsonl 
```



## FAQ
When adding a new datasets, you probably will encounter following questions:

* #### what if existing task schema can not support my current dataset?
Suggested docs: [how to add_new_task_schema](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_task_schema.md)

* #### how to add the language information of my dataset?
Suggested doc: [how to add language information](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_language_info.md)

* #### could I have more examples to help me write the script?
You can find scripts for adding different datasets cross different tasks.
Suggested doc: [more examples](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/task_normalization.md)

For example,
   * if you aim to add a simple text classification dataset:
       * [sst2](https://github.com/ExpressAI/DataLab/blob/main/datasets/sst2/sst2.py)
   * if you aim to add a simple summarization dataset
       * [cnn_dailymail](https://github.com/ExpressAI/DataLab/blob/main/datasets/cnn_dailymail/cnn_dailymail.py)
   * if you aim to add a simple natural languange inference dataset:
       * [sick](https://github.com/ExpressAI/DataLab/blob/main/datasets/sick/sick.py)
       * [snli](https://github.com/ExpressAI/DataLab/blob/main/datasets/snli/snli.py)
   * if you aim to add a datasets with different versions/domains/languages/subdatasets
       * different versions: [arxiv_sum](https://github.com/ExpressAI/DataLab/blob/main/datasets/arxiv_sum/arxiv_sum.py)
       * different languages: [xlsum](https://github.com/ExpressAI/DataLab/blob/main/datasets/xlsum/xlsum.py), [mlqa](https://github.com/ExpressAI/DataLab/blob/main/datasets/mlqa/mlqa.py)
       * different subtasks: [glue](https://github.com/ExpressAI/DataLab/blob/main/datasets/glue/glue.py)

   * if your datasets have been packaged into a zip file, you can refer to this [example](https://github.com/ExpressAI/DataLab/blob/main/datasets/snli/snli.py)

   * if you want to upload your dataset into DataLab web platform (which provides a bunch of data visualization and analysis), you can follow
   this [doc](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_web_platform.md).

NOTE:
* Usually, using the **Lower** case string with `_` instead of `-` for the script name (`arxiv_sum.py`) while camel case for the class name (`ArxivSum`).
   
#### what if there is private data that I want to use?

DataLab has a special environmental variable `DATALAB_PRIVATE_LOC` that you can use to
store private datasets. It can be a web location or a location on your filesystem.
Insert this exact string `DATALAB_PRIVATE_LOC` into your dataset location, and then
set an environmental variable:

    export DATALAB_PRIVATE_LOC=/path/to/private/root

and the environmental variable will be substituted into your dataset path. You can
seen an example of how this is done in the
[fig_qa dataloader](https://github.com/ExpressAI/DataLab/blob/main/datasets/fig_qa/fig_qa.py).
