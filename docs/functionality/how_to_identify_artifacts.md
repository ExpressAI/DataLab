# How to Identify Dataset Artifacts Using DataLab?


The basic idea of artifact identification is to use PMI (Pointwise mutual information) to detect whether there
is an association between **TWO** features (e.g., sentence length v.s category).

For example, given two feature_i, and feature_j, a higher absolute value of PMI(feature_i, feature_j) 
suggests: 
* higher association between feature_i, and feature_j; 
* a potential artifact pattern involving feature_i, and feature_j, (e.g., longer sentences tend to have a positive sentiment.)


One famous example of examining artifacts from previous work is [Annotation Artifacts in Natural Language Inference Data](https://arxiv.org/pdf/1803.02324.pdf).
Inspired by this work, we take [`snli`](http://datalab.nlpedia.ai/normal_dataset/617794bfb7314cb4146d2384/dataset_bias) dataset 
as one example and walk through how to identify potential artifacts of a dataset using DataLab.
 


### Step 1: Find the dataset's artifact analysis page
Navigate to the dataset by either (1) searching for it in the top bar, or (2) clicking the "Data" button on the front page and browsing to it.
Click the `Bias` tab, and then under the "choose bias dimension" menu select `artifact identification`.

![image](https://user-images.githubusercontent.com/59123869/154880168-d6def7f6-2833-4665-a490-3cb09fd199d4.png)


### Step 2: Select two feature fields

Before selecting two specific features, we first select a `field`.
Here by `field`, we indicate basic piece of information in the example.
For example, in natural language inference task, the `field` could be `premise`, `hypothesis`, and `label`.
In text summarization task, the `field` could be `source` and `summary`.


### Step 3: Select two features
Features are properties defined over the data of each field. For example,
the `text length` could be a feature of the field `hypothesis`, or `label` itself could be a feature of the field `label`.

![image](https://user-images.githubusercontent.com/59123869/154881277-c1b1de9a-3a07-4446-8a2b-e9fbad161d75.png)



### Step 4: Interpret results
Once we finished the above three steps, a visualized PMI matrix will be printed automatically:
![image](https://user-images.githubusercontent.com/59123869/154881323-0e614f18-571d-4340-b991-91e5b35f3f80.png)


For example, in the above matrix, value in entry (i,j) represents the PMI(i,j).
and `8.4~12.1 (count:10883)` shows that there are 10883 samples whose hypothesis lengths are in [8.4,12.1].


One tip here is we can examine the entries with higher absolute value of PMI (i.e., darker color), suggesting a potential artifact pattern.

For each entry, you also can see a floating text box. For example, in this example, suppose that
x = `hypothesis length` satisfies [8.4, 12.1],
y = `category label` is neutral
N = `the number of all samples`: 50000

we have:

* p(x) = n(x)/N = 10883/50000  (n(x))
* p(y) = n(x)/N = 16525/50000
where n(x) represents the number of samples with feature x.
We define
* `actual number` as the actual number of samples with features x and y: N * P(x,y) 
* `ideal count` as the number of samples with features x and y if they were independent: N * P(x) * P(y)
* `PMI` = log p(x,y)/(p(x)p(y))

Then for each entry of the matrix, we can see one floating text box, showing following statistics:
* `label` (neutral): one feature
* `hypothesis_length` (8.4~12.1): another feature
* `actual count`: the actual number of samples whose `label` is neutral and `hypothesis_length` is in [8.4~12.1].
* `ideal count`: the ideal number of samples whose `label` is neutral and `hypothesis_length` is in [8.4~12.1].
* `overrepresented`: the ratio between `actual count` and `ideal count`.
* `PMI`: the PMI value  

From this PMI matrix, we can observe that:
* when the length of hypothesis is larger than 8.4, PMI(label_neutral,length_hypothesis) >0.28, suggesting that "long hypotheses" tend to co-occur with 
  the "neutral" label regardless of what the premises are.
* when length_hypothesis ∈ [1,4.7], PMI(label_entailment,length_hypothesis) = 0.359, implying that "short hypotheses" tend to co-occur 
  with the label "entailment".

The above is just one example, and we can identify more potential artifacts in a similar way based on different features provided by DataLab.
