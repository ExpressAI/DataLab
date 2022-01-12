# How to add new datasets?

We will walk through how to add a new dataset into datalab.


## 1. Clouding your raw dataset
Put your dataset into a server with downloadable links.
For example, you can place your datasets in this shared gdrive [folder](https://drive.google.com/drive/folders/1JttBMEoUmVZ8wF7Qa6C8h32XJpqEOd7u?usp=sharing)


## 2. Get the downloadable url for datasets

if your link is from google drive, you just need to modify following template by replacing `ID` with real string

`https://drive.google.com/uc?id=ID=download`

You can get `ID` from the link of `sharing to any`, for example, we can know
`ID` is: `1t-2aRCGru5yJzpJ-o4uB6UmHbNRzNfIb` from 

`https://drive.google.com/file/d/1t-2aRCGru5yJzpJ-o4uB6UmHbNRzNfIb/view?usp=sharing`, 
so finally, we have

`https://drive.google.com/uc?id=1t-2aRCGru5yJzpJ-o4uB6UmHbNRzNfIb=download`


## 3. Creat a new folder and write a config python script inside it.

Suppose the dataset name to be added is `ag_news`, we need to:
* creat a folder `ag_news` in [DataLab/datasets/](https://github.com/ExpressAI/DataLab/tree/main/datasets)
* creat a config script `ag_news.py` in the above folder, i.e., `Datalab/datasets/ag_news/ag_news.py`
* finish the config script based on some provided examples:
    * text-classification: [template](https://github.com/ExpressAI/DataLab/blob/main/datasets/ag_news/ag_news.py)
    


## 4. Test in your local server
* enter into `Datalab/datasets` folder
* run following python command

```python
   from datalabs import load_dataset
   dataset = load_dataset("./ag_news")
   print(dataset['train']._info)
```


