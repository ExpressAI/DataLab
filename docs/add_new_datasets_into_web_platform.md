# Add New Datasets into the DataLab Web Platform

DataLab provides an API which users can use to upload their own datasets into the DataLab web platform.
There are following specific situations:


## 1. quickly add metadata information of a new dataset
This is useful when you just want to quickly add a new record of your new dataset into the database without uploading any 
data samples.

```python

client = Client(dataset_name_db="test_dataset")
client.add_dataset_metadata()
```
where `dataset_name_db` denotes the name of dataset to be stored in the database.
You can specify more metadata information by:

```python

client = Client(dataset_name_db="test_dataset",
                 version = "origin",
                 languages = ['en'],
                 tasks = ['text-classification'],
                 task_categories = ['text-classification'],)
client.add_dataset_metadata()
```


## 2. Add new datasets based on implemented data loader scripts of the SDK without calculating additional features
This is useful when:
* you have already implemented a data loader scripts of the new dataset [here](https://github.com/ExpressAI/DataLab/tree/main/datasets).
* you just want to add both the metadata and sample information into the database
* you don't aim to calculate the additional features


Note: so far, we only upload the first 20,000 samples into the web database.

This involves two steps:

#### (1) write a data loader script for your new dataset. You can refer to [`how to add new datasets into sdk`](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_datasets.md)

#### (2) call the client function to add your dataset

```python
client = Client(dataset_name_db="test_dataset2", dataset_name_sdk="qc")
client.add_dataset_from_sdk()
```
where
* `dataset_name_db`: denotes the name of dataset to be stored in the database.
* `dataset_name_sdk`: denotes the data loader script's name of your dataset.




## 3. Add new datasets based on implemented data loader scripts of the SDK with calculated additional features
This is useful when:
* you have already implemented a data loader scripts of the new dataset [here](https://github.com/ExpressAI/DataLab/tree/main/datasets).
* you just want to add both the metadata and sample information into the database
* you plan to calculate the additional features

Note: so far, we only upload the first 20,000 samples into the web database.



This involves two steps:

#### (1) write a data loader script for your new dataset. You can refer to [`how to add new datasets into sdk`](https://github.com/ExpressAI/DataLab/blob/main/docs/add_new_datasets.md)

#### (2) call the client function to add your dataset

```python
client = Client(dataset_name_db="test_mr", dataset_name_sdk="mr", \
                calculate_features = True, 
                field = "text",
                data_typology = 'textdataset')
client.add_dataset_from_sdk()
```
where
* `dataset_name_db`: denotes the name of dataset to be stored in the database.
* `dataset_name_sdk`: denotes the data loader script's name of your dataset.
* `calculate_features = True`: refers to additional features will be calculated (e.g., `text length`)
* `field`: a field that features (e.g., `text length`) will be extracted from. It is usually task (or even dataset) dependent. We will give more docs about how to set it.
* `data_typology`: the typology name of the dataset (We will give more explanation about this.)