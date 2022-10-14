# Analyzing Summarization Datasets

Datalab supports fine-grained analysis and comparision of summarization datasets. Please refer to [fine-grained-analysis](2_fine_grained_analysis_of_a_dataset.md) and [compare-two-datasets](7_compare_two_datasets.md) for more usage details.

## Outline

* ### [Supported Features](#supported-features-1)

* ### [Supported Datasets](#supported-datasets-1)

* ### [Examples](#examples-1)

* #### [CNNDM v.s. XSum](#comparing-cnndm-cnndailymail-and-xsum-datasets)

* #### [Multi-XScience v.s. DialogSum](#comparing-a-scientific-multi-document-summarization-dataset-multi-xscience-and-a-dialogue-summarization-dataset-dialogsum)

* #### [Reddit TIFU v.s. QMSum](#comparing-a-summarization-dataset-reddit-tifu-on-informal-crowd-generated-posts-and-a-query-based-meeting-summarization-dataset-qmsum)

## Supported Features

### General Features

* text length
* ratio of basic words
* lexcical richness (i.e. lexical diversity)
* ...

### Advanced Task-Specific Features

* Density (Extractive Fragment Density)
  * Introduced in [Grusky et al., 2018](https://aclanthology.org/N18-1065.pdf)
  * The density measure quantifies how well the word sequence of a summary can be described as a series of extractions.
  * DENSITY(A, S) is the average length of the extractive fragment to which each word in the summary belongs.
* Coverage (Extractive Fragment Coverage)
  * Introduced in [Grusky et al., 2018](https://aclanthology.org/N18-1065.pdf)
  * The coverage measure quantifies the extent to which a summary is derivative of a text.
  * COVERAGE(A, S) measures the percentage of words in the summary that are part of an extractive fragment with the article.
* Compression (Compression Ratio)
  * Introduced in [Grusky et al., 2018](https://aclanthology.org/N18-1065.pdf)
  * Summarizing with higher compression is challenging as it requires capturing more precisely the critical aspects of the article text.
  * COMPRESSION is the word ratio between the article and summary.
* Repetition
  * The repetition ratio of the reference summaries.
  * A large reprtition ratio may indicate redundency.
  * Please refer to [See et al., 2017](https://aclanthology.org/P17-1099.pdf) for more details.
* Novelty
  * The ratio of content of reference summaries that are not in the source text.
  * Novelty is correlated with both abstractivity and faithfulness.
  * Please refer to [See et al., 2017](https://aclanthology.org/P17-1099.pdf), [Bommasani and Cardie, 2020](https://aclanthology.org/2020.emnlp-main.649.pdf), [Maynez et al., 2020](https://aclanthology.org/2020.acl-main.173.pdf) for more details.
* Copy Length
  * The average length of fragments of reference summaries that appear in the source text.
  * Please refer to [Chen et al., 2020](https://aclanthology.org/2020.findings-emnlp.329.pdf), [Grusky et al., 2018](https://aclanthology.org/N18-1065.pdf) for more details.

## Supported Datasets

|Datasets|Version|Task Schema|Dataloader|Comments
|:---    |:---        |:---       |:--- |:---  
govreport | -  | Summarization | ``` load_dataset("govreport") ``` | Current definition: text, summary |
dialogsum | `document`  | Summarization | ``` load_dataset("dialogsum", "document") ``` | Current definition: text, summary |
dialogsum | `dialogue`  | DialogSummarization | ``` load_dataset("dialogsum", "dialogue") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]` |
wikihow | -  | Summarization | ``` load_dataset("wikihow") ``` | Current definition: text, summary |
wikisum | -  | Summarization | ``` load_dataset("wikisum") ``` | Current definition: text, summary |
reddit_tifu | -  | Summarization | ``` load_dataset("reddit_tifu") ``` | Current definition: text, summary |
bigpatent | -  | Summarization | ``` load_dataset("bigpatent") ``` | Current definition: text, summary |
multi_xscience | `single-document`  | Summarization | ``` load_dataset("multi_xscience", "single-document") ``` | Current definition: text, summary |
multi_xscience | `multi-document`  | MultiDocSummarization | ``` load_dataset("multi_xscience", "multi-document") ``` | Current definition: texts: `List[str]`, summary: `str`|
multinews | `raw-single`  | Summarization | ``` load_dataset("multinews", "raw-single") ``` | raw data, Current definition: text, summary |
multinews | `raw-cleaned-single`  | Summarization | ``` load_dataset("multinews", "raw-cleaned-single") ``` | cleaned raw data, Current definition: text, summary |
multinews | `preprocessed-single`  | Summarization | ``` load_dataset("multinews", "preprocessed-single") ``` | preprocessed data, Current definition: text, summary |
multinews | `truncated-single`  | Summarization | ``` load_dataset("multinews", "truncated-single") ``` | preprocessed and truncated data, Current definition: text, summary |
multinews | `raw-multi`  | MultiDocSummarization | ``` load_dataset("multinews", "raw-multi") ``` | raw data, Current definition: texts: `List[str]`, summary: `str` |
multinews | `raw-cleaned-multi`  | MultiDocSummarization | ``` load_dataset("multinews", "raw-cleaned-multi") ``` | cleaned raw data, Current definition: texts: `List[str]`, summary: `str` |
multinews | `preprocessed-multi`  | MultiDocSummarization | ``` load_dataset("multinews", "preprocessed-multi") ``` | preprocessed data, Current definition: texts: `List[str]`, summary: `str` |
multinews | `truncated-multi`  | MultiDocSummarization | ``` load_dataset("multinews", "truncated-multi") ``` | preprocessed and truncated data, Current definition: texts: `List[str]`, summary: `str` |
samsum | `document`  | Summarization | ``` load_dataset("samsum", "document") ``` | Current definition: text, summary |
samsum | `dialogue`  | DialogSummarization | ``` load_dataset("samsum", "dialogue") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]` |
qmsum | `document`  | Summarization | ``` load_dataset("qmsum", "document") ``` | Current definition: text, summary |
qmsum | `query-based`  | QuerySummarization | ``` load_dataset("qmsum", "query-based") ``` | Current definition: text, summary, query |
nlcs | `en2zh` | Summarization | ``` load_dataset("nlcs", "en2zh") ``` | Current definition: text, summary |
cnewsum | `document` | Summarization | ``` load_dataset("cnewsum") ``` | Current definition: text, summary |
csds | `document` | Summarization | ``` load_dataset("csds", "document") ``` | Current definition: text, summary |
csds | `usersumm` | DialogSummarization | ``` load_dataset("csds", "usersumm") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]`  |
csds | `agentsumm` | DialogSummarization | ``` load_dataset("csds", "agentsumm") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]` |
csds | `finalsumm` | DialogSummarization | ``` load_dataset("csds", "finalsumm") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]` |
nctb | `document` | Summarization | ``` load_dataset("nctb") ``` | Current definition: text, summary |
gamewikisum | `document` | Summarization | ``` load_dataset("gamewikisum") ``` | Current definition: text, summary |
gamewikisum | `multidoc` | MultiDocSummarization | ``` load_dataset("gamewikisum", "multidoc") ``` | Current definition: texts: `List[str]`, summary: `str` |
ssn | `transductive-document` | Summarization | ``` load_dataset("ssn", "transductive-document") ``` | Current definition: text, summary |
ssn | `inductive-document` | Summarization | ``` load_dataset("ssn", "inductive-document") ``` | Current definition: text, summary |
ssn | `transductive-multidoc` | MultiDocSummarization | ``` load_dataset("ssn", "transductive-multidoc") ``` | Current definition: texts: `{"introduction": str, "references": List[str]}`, summary: `str` |
ssn | `inductive-multidoc` | MultiDocSummarization | ``` load_dataset("ssn", "inductive-multidoc") ``` | Current definition: texts: `{"introduction": str, "references": List[str]}`, summary: `str` |
pn_summary | `document` | Summarization | ``` load_dataset("pn_summary") ``` | Current definition: text, summary |
scitldr | `tldr-auth` | Summarization | ``` load_dataset("scitldr", "tldr-auth") ``` | Current definition: text: `str`, summary: `str` |
scitldr | `tldr-pr` | Summarization | ``` load_dataset("scitldr", "tldr-pr") ``` | Current definition: text: `str`, summary: `List[str]` |
gazeta | `document` |  Summarization | ``` load_dataset("gazeta") ``` | Current definition: text, summary |
mlgsum | 12 languages: `["de", "en", "es", "fr", "hi", "id", "pt", "ru", "tr", "uk", "vi", "zh"]` |  Summarization | ``` load_dataset("mlgsum", "LANGUAGE_ID") ``` | Current definition: `text`, `summary`, Multilingual Dataset |
wikilingua | 18 languages: `["en", "es", "pt", "fr", "de", "ru", "it", "id", "nl", "ar", "zh", "vi", "th", "ja", "ko", "hi", "cs", "tr"]` |  Summarization | ``` load_dataset("wikilingua", LANGUAGE_ID) ``` | Current definition: `text`, `summary`, Multilingual Dataset |
wikilingua | 17 language pairs: `["es-en", "pt-en", "fr-en", "de-en", "ru-en", "it-en", "id-en", "nl-en", "ar-en", "zh-en", "vi-en", "th-en", "ja-en", "ko-en", "hi-en", "cs-en", "tr-en"]` |  Summarization | ``` load_dataset("wikilingua", "LANGUAGE_ID-en") ``` | Current definition: `text`, `summary`, Crosslingual Dataset |
qbsum | `query-based`  | QuerySummarization | ``` load_dataset("qbsum", "query-based") ``` | Current definition: `text`, `summary`, `query` |
open4business | `document`  | Summarization | ``` load_dataset("open4business", "document") ``` | Current definition: `text`, `summary` |
summscreen | `non-anonymized-ForeverDreaming`  | Summarization | ``` load_dataset("summscreen", "non-anonymized-ForeverDreaming") ``` | Current definition: `text`, `summary`, ForeverDreaming split, non-anonymized version, tokenized |
summscreen | `anonymized-ForeverDreaming`  | Summarization | ``` load_dataset("summscreen", "anonymized-ForeverDreaming") ``` | Current definition: `text`, `summary`, ForeverDreaming split, anonymized version, tokenized |
summscreen | `non-anonymized-TVMegaSite`  | Summarization | ``` load_dataset("summscreen", "non-anonymized-TVMegaSite") ``` | Current definition: `text`, `summary`, TVMegaSite split, non-anonymized version, tokenized |
summscreen | `anonymized-TVMegaSite`  | Summarization | ``` load_dataset("summscreen", "anonymized-TVMegaSite") ``` | Current definition: `text`, `summary`, TVMegaSite split, anonymized version, tokenized |
aeslc | `document` | Summarization | ```load_dataset("aeslc")``` | Current definition: text, summary |
k_sportssum | `document` | Summarization | ```load_dataset("k_sportssum")``` | Current definition: text: `{"commentary": List[str], "score": List[str]}`, summary: `str` |
peersum | `document` | Summarization | ```load_dataset("peersum")``` | Current definition: text, summary |
peersum | `dialogue` | DialogSummarization | ``` load_dataset("peersum", "dialogue") ``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `List[str]`  |
pts | `document` | Summarization | ```load_dataset("pts")``` | Current definition: text, summary |
amasum | `verdict-ref` | MultiDocSummarization | ```load_dataset("amasum", "verdict-ref")``` | Current definition: texts: `List[str]`, summary: `str` |
amasum | `pros-ref` | MultiDocSummarization | ```load_dataset("amasum", "pros-ref")``` | Current definition: texts: `List[str]`, summary: `str` |
amasum | `cons-ref` | MultiDocSummarization | ```load_dataset("amasum", "cons-ref")``` | Current definition: texts: `List[str]`, summary: `str` |
ami | `document` |  Summarization | ```load_dataset("ami", "document")``` | Current definition: text: `str`, summary: `str` |
ami | `dialogue` | DialogSummarization |  ```load_dataset("ami", "dialogue")``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `str`  |
icsi | `document` |  Summarization | ```load_dataset("icsi", "document")``` | Current definition: text: `str`, summary: `str` |
icsi | `dialogue` | DialogSummarization | ```load_dataset("icsi", "dialogue")``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `str`  |
citesum | `document` |  Summarization | ```load_dataset("citesum")``` | Current definition: text: `str`, summary: `str` |
klexikon | `document` |  Summarization | ```load_dataset("klexikon")``` | Current definition: text: `str`, summary: `str` |
streamhover | `document` | Summarization | ```load_dataset("streamhover")``` | Current definition: text: `str`, summary: `str` |
welsh | `wiki-ref` |Summarization | ```load_dataset("welsh", "wiki-ref")``` | Current definition: text: `str`, summary: `str` |
welsh | `human-ref` | MultiRefSummarization | ```load_dataset("welsh", "human-ref")``` | Current definition: text: `str`, summaries: `List[str]` |
wsd | `monolingual`| Summarization | ```load_dataset("wsd", "monolingual")``` | Current definition: text: `str`, summary: `str` |
wsd | `crosslingual` | Summarization | ```load_dataset("wsd", "crosslingual")``` | Current definition: text: `str`, summary: `str` |
crosssum | 45 languages X 45 languages paris, e.g. `en-en`, langauges: `["am", "ar", "az", "bn", "my", "zh-CN", "zh-TW", "en", "fr", "gu", "ha", "hi", "ig", "id", "ja", "rn", "ko", "ky", "mr", "np", "om", "ps", "fa", "pcm", "pt", "pa", "ru", "gd", "sr-C", "sr-L", "si", "so", "es", "sw", "ta", "te", "th", "ti", "tr", "uk", "ur", "uz", "vi", "cy", "yo"]` | `Summarization` | ```load_dataset("crosssum", language_paris)``` | Current definition: text: `str`, summary: `str` |
mediasum | `dialogue` | DialogSummarization | ```load_dataset("mediasum", "dialogue")``` | Current definition: dialogue: `{"speaker": List[str], "text": List[str]}`, summary: `str` |
mediasum | `document` | Summarization | ```load_dataset("mediasum", "document")``` | Current definition: text: `str`, summary: `str` |
tweetsum | `document` | Summarization | ```load_dataset("tweetsum")``` | Current definition: text: `str`, summary: `str` |
shortstories | `abstractive` | Summarization | ```load_dataset("shortstories", "abstractive")``` |  Current definition: text: `str`, summary: `str` |
shortstories | `extractive-smmry` | ExtractiveSummarization | ```load_dataset("shortstories", "extractive-smmry")``` |  Current definition: text: `str`, summary: `List[str]` |
shortstories | `extractive-resoomer` | ExtractiveSummarization | ```load_dataset("shortstories", "extractive-resoomer")``` |  Current definition: text: `str`, summary: `List[str]` |
sumpubmed | `document` | Summarization | ```load_dataset("sumpubmed")``` | Current definition: text: `str`, summary: `str` |
answersumm | `query-multi-doc` | QueryMultiDocSummarization | ```load_dataset("answersumm")``` | Current definition: query: `str` texts: `List[str]`, summary: `str` |
vt-ssum | `document` | ExtractiveSummarization | ```load_dataset("vt-ssum")``` |  Current definition: text: `str`, summary: `List[str]` |
wikicite | `query-multi-doc` | QueryMultiDocSummarization | ```load_dataset("wikicite")``` | Current definition: query: `str` texts: `List[str]`, summary: `str` |
scisummnet | `multidoc` | MultiDocSummarization | ```load_dataset("scisummnet", "multidoc")``` |  Current definition: texts: `List[str]`, summary: `str` |
scisummnet | `document` | Summarization | ```load_dataset("scisummnet", "document")``` | Current definition: text: `str`, summary: `str` |
wikihowqa | `document` | Summarization | ```load_dataset("wikihowqa")``` | Current definition: text: `str`, summary: `str` |
weibo_summary_comments | `document` | ReaderAwareSummarization | ```load_dataset("weibo_summary_comments")``` | Current definition: text: `str` comments: `List[str]`, summary: `str` |
debatesum | `extract` | Summarization | ```load_dataset("debatesum", "extract")``` | Current definition: text: `str`, summary: `str` |
debatesum | `abstract` | Summarization | ```load_dataset("debatesum", "abstract")``` | Current definition: text: `str`, summary: `str` |
multilexsum | `long` | MultiDocSummarization | ```load_dataset("multilexsum", "long")``` |  Current definition: texts: `List[str]`, summary: `str` |
multilexsum | `short` | MultiDocSummarization | ```load_dataset("multilexsum", "short")``` |  Current definition: texts: `List[str]`, summary: `str` |
multilexsum | `tiny` | MultiDocSummarization | ```load_dataset("multilexsum", "tiny")``` |  Current definition: texts: `List[str]`, summary: `str` |
20minuten | `document` | Summarization | ```load_dataset("20minuten")``` | Current definition: text: `str`, summary: `str` |

## Examples

### Comparing CNNDM (CNN/DailyMail) and XSum datasets

<img src="https://user-images.githubusercontent.com/51046084/155863520-5f08e2da-f791-47c3-95d6-a8802fb7e671.JPG" width="600"/>

#### Advanced Features

<img src="https://user-images.githubusercontent.com/51046084/155863546-e9b16929-629b-4e43-b61d-3822d4642d96.JPG" width="600"/>

<!-- ![advance](https://user-images.githubusercontent.com/51046084/155863546-e9b16929-629b-4e43-b61d-3822d4642d96.JPG) -->

Compared with CNNDailyMail, XSum is more abstractive since it achieves high novelty and compression rate, and much lower density and coverage.

### Comparing a scientific multi-document summarization dataset (Multi-XScience) and a dialogue summarization dataset (DialogSum)

<!-- ![dataset](https://user-images.githubusercontent.com/51046084/155858814-11938e42-b4aa-473e-947e-07521fd25c5e.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155858814-11938e42-b4aa-473e-947e-07521fd25c5e.JPG" width="600"/>

#### General Features

<!-- ![general](https://user-images.githubusercontent.com/51046084/155858866-d3a5eafe-a94e-4804-9a85-f99f5a4ad719.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155858866-d3a5eafe-a94e-4804-9a85-f99f5a4ad719.JPG" width="600"/>

* Multi-XScience is longer than DialogSum in terms of both text length and summary length.
* DialogSum contain more basic words than Multi-XScience, possibly because dialogues are from daily life while scientific papers contain more rare words.

#### Advanced Features
<!-- ![advance](https://user-images.githubusercontent.com/51046084/155859135-b85350c9-4e47-4cc3-b6f4-82e8dfc71e6d.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155859135-b85350c9-4e47-4cc3-b6f4-82e8dfc71e6d.JPG" width="600"/>

* Multi-Science is more abstractive since it contains more novel words
* DialogSum is more extractive since it has higher density, coverage and copy length, which means more content in the summaries can be extracted from the source text.

### Comparing a summarization dataset (Reddit TIFU) on informal crowd-generated posts and a query-based meeting summarization dataset (QMSum)

<!-- ![name](https://user-images.githubusercontent.com/51046084/155859674-e3c805a7-c753-4838-a32b-d549077a5dac.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155859674-e3c805a7-c753-4838-a32b-d549077a5dac.JPG" width="600"/>

#### General Features

<!-- ![general](https://user-images.githubusercontent.com/51046084/155859717-e27c3588-308b-49d2-b901-fd570583b16d.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155859717-e27c3588-308b-49d2-b901-fd570583b16d.JPG" width="600"/>

A salient difference between these two datasets is that QMSum is much longer than Reddit TIFU. As QMSum is a query-based dataset, its source text may contain infromation about multiple queries and their associated summaries, which results in longer source text.

#### Advanced Features
<!-- ![advance](https://user-images.githubusercontent.com/51046084/155859135-b85350c9-4e47-4cc3-b6f4-82e8dfc71e6d.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155859951-4cce92ed-0b12-4db3-89b6-77031b519491.JPG" width="600"/>

<!-- ![advance](https://user-images.githubusercontent.com/51046084/155859951-4cce92ed-0b12-4db3-89b6-77031b519491.JPG) -->

The advanced features indicate that QMSum is more extractive while Reddit TIFU contains more novel words. As a result, it is likely that QMSum's summaries are more faithful than Reddit, since more content of QMSum's summaries can be direcly recovered in the source test. This observation suggests that the query-based collection protocal of QMSum can improve the faithfulness of reference summaries, which is one of the goals of this dataset.
