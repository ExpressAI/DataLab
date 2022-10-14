# How to Get more Fine-grained Information of a Dataset?

Fine-grained analysis aims to answer the question: **what are the characteristics of a dataset?**
Conceptually, each data point (i.e. sample-level) or whole dataset (i.e. dataset-level) can be characterized from different dimensions.
These are either generic (`text length` at sample-level or `the average text length` at corpus-level) or task-specific
(for summarization: `summary compression` `the average of summary compression`)

One key contribution of DataLab is that we not only design rich sample-level and dataset-level features,
but also compute and store those features in a database for easy browsing. For example, so far, we have designed more than 300 features
and computed features for 140M samples.

## Dataset-level Analysis

### 1. Choose a dataset and click the `overview` button

<img src="https://user-images.githubusercontent.com/59123869/155372937-3400e723-b302-4fa4-96a4-e39825073e75.png" width="200"/>

### 2. You can see a bunch of dataset-level information that DataLab generates for you

<img src="https://user-images.githubusercontent.com/59123869/155373748-5ce1111d-907f-4501-9c00-539d3a4f6581.png" width="600"/>

```
when writing a paper, we usually need some table like this, using DataLab, you can make your table more comprehensive!
```

<img src="https://user-images.githubusercontent.com/59123869/155373996-a4504ee2-0044-4c46-911b-117824696bca.png" width="600"/>

### 3. Make your contribution

DataLab has devised a comprehensive schema for each dataset based on [Data Statement](https://aclanthology.org/Q18-1041/), [LREC Database](https://lrec2020.lrec-conf.org/en/shared-lrs/), [Huggingface](https://huggingface.co/docs/datasets/), and [Paperswithcode](https://paperswithcode.com/).
Regarding some important information that required community wisdom, DataLab is positioned as a crowdsourceable platform and any researcher can contribute to by directly
editting the form.
For example:

<img src="https://user-images.githubusercontent.com/59123869/155375250-17772f8c-ff32-45a0-8b96-353b03848241.png" width="600"/>

## Sample-level Analysis

### 1. Choose a dataset and click the `sample` button

<img src="https://user-images.githubusercontent.com/59123869/155375407-be78af20-5214-4ff5-a50f-a9de5ba0400a.png" width="200"/>

### 2. Filter samples based on different features

You can filter data samples based on different features. **Dont' forget click the `Confirm` button once you finalize some feature.**

<img src="https://user-images.githubusercontent.com/59123869/155375881-9a43adc7-80dc-4a01-b730-31388690cd08.png" width="600"/>

<img src="https://user-images.githubusercontent.com/59123869/155375992-385d37d1-1e17-455b-a4ad-b0195a22b8a9.png" width="200"/>

### 3. Browse samples

In the middle of the page, you can examine detailed sample-level information of your filtered sample (the raw text together with a bunch of features, such as `text length`).

### 4. Analyze by Sample Distribution

One cool thing that DataLab has done for you is to automatically generate a sample distribution based on different features that you're interested in.
For example, the following chart shows the sample distribution over different text lengths.

<img src="https://user-images.githubusercontent.com/59123869/155376801-b9b75821-9282-43b1-a09f-e795f462a52e.png" width="600"/>

You can choose more features:

<img src="https://user-images.githubusercontent.com/59123869/155376959-53b638f7-6b20-4a09-b22e-43f4bc0a128a.png" width="600"/>
