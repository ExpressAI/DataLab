# How to Perform Hate Speech Analysis using DataLab



DataLab supports different types of bias analyses for datasets and `hate speech` is one case.
Specifically, given a dataset, DataLab can identify what percentage of samples contains hate speech words.
Although deciding whether a sentence contains toxic language is a slightly complex task, which may involve the confounding effects of 
dialect and the social identity of a speaker [Sap et al. (2019)](https://aclanthology.org/P19-1163.pdf), we make a first step by following [Davidson et al (2017)](https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665), 
classifying the samples into following categories:
* `hate speech`
* `offensive language`
* `neither`

We then calculate the ratio of samples in different categories.


To perform this type of analysis:

### 1. Dataset Selection
You just need to choose a dataset and click the right mouse button, and choose `analysis` -> `bias`, then you will enter into a page designed for bias analysis

<img src="https://user-images.githubusercontent.com/59123869/155384702-9c7dc15b-036f-4ce4-906d-1258075dad8a.png" width="200">


 

### 2. Choose the `hate speech` filter

As shown below, different colors represent the proportions of samples with different categories (`hate speech`, `offensive language` and `neither`)

<img src="https://user-images.githubusercontent.com/59123869/155385027-17d4246b-2551-4ce2-9d31-b6305433ad08.png" width="600">


 







