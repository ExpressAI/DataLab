# How to add new datasets?

We will walk through how to add a new dataset into datalab.


## 1. Clouding your raw dataset
Put your dataset into a server with downloadable links.
For example, you can place your datasets in gdrive [folder](https://drive.google.com/drive/folders/1JttBMEoUmVZ8wF7Qa6C8h32XJpqEOd7u?usp=sharing) (But you don't need to put your data here since this is just one example.)


## 2. Get the downloadable url for datasets

if your link is from google drive, you need to modify the following template by replacing `FILEID` with real string

`https://drive.google.com/uc?export=download&id=FILEID`

You can get `FILEID` from the link of `sharing to any`, for example, we can know
`FILEID` is: `1JX8pdQJaDqwzK7fzNs9mM9UY09be29ci` from 

`https://drive.google.com/file/d/1JX8pdQJaDqwzK7fzNs9mM9UY09be29ci/view?usp=sharing`, 
so finally, we have

`https://drive.google.com/uc?export=download&id=1JX8pdQJaDqwzK7fzNs9mM9UY09be29ci`


## 3. Create a new folder and write a config python script inside it.

Suppose the dataset name to be added is `cr`, we need to:
* create a folder `cr` in [DataLab/datasets/](https://github.com/ExpressAI/DataLab/tree/main/datasets)
* create a config script `cr.py` in the above folder, i.e., `Datalab/datasets/cr/cr.py`
* finish the config script based on some provided examples:
    * text-classification: [template](https://github.com/ExpressAI/DataLab/tree/main/datasets/cr)
    * extractive-qa: [template](https://github.com/ExpressAI/DataLab/blob/main/datasets/squad/squad.py)
    


## 4. Test in your local server
* enter into `Datalab/datasets` folder
* run following python command

```python
   from datalabs import load_dataset
   dataset = load_dataset("./cr")
   print(dataset['train']._info)
   print(dataset['train']._info.task_templates)
```

## 5. Update your updated information of your dataset
Once you successfully add a new dataset, please update the [table](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/task_normalization.md).


## FAQ
When adding a new datasets, you probably will encounter following questions:

* #### what if existing task schema can not support my current dataset?
Suggested docs: [how to add_new_task_schema](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_task_schema.md)

* ### how to add the language information of my dataset?
Suggested doc: [how to add language information](https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_language_info.md)


NOTE:
* Usually, using the lower case string for the script name (arxiv_sum.py) while camel case for the class name (`ArxivSum`).
