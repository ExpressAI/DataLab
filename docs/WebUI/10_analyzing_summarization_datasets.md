# Analyzing Summarization Datasets


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
multi_xscience | `single-document`  | Summarization | ``` load_dataset("multi_xsience", "single-document") ``` | Current definition: text, summary |
multi_xscience | `multi-document`  | MultiDocSummarization | ``` load_dataset("multi_xsience", "multi-document") ``` | Current definition: texts: `List[str]`, summary: `str`|
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

## Examples

### Comparing a scientific multi-document summarization dataset (Multi-XScience) and a dialogue summarization dataset (DialogSum)

<!-- ![dataset](https://user-images.githubusercontent.com/51046084/155858814-11938e42-b4aa-473e-947e-07521fd25c5e.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155858814-11938e42-b4aa-473e-947e-07521fd25c5e.JPG" width="600"/>

#### General Features

<!-- ![general](https://user-images.githubusercontent.com/51046084/155858866-d3a5eafe-a94e-4804-9a85-f99f5a4ad719.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155858866-d3a5eafe-a94e-4804-9a85-f99f5a4ad719.JPG" width="600"/>


- Multi-XScience is longer than DialogSum in terms of both text length and summary length.
- DialogSum contain more basic words than Multi-XScience, possibly because dialogues are from daily life while scientific papers contain more rare words.

#### Advanced Features
<!-- ![advance](https://user-images.githubusercontent.com/51046084/155859135-b85350c9-4e47-4cc3-b6f4-82e8dfc71e6d.JPG) -->

<img src="https://user-images.githubusercontent.com/51046084/155859135-b85350c9-4e47-4cc3-b6f4-82e8dfc71e6d.JPG" width="600"/>

- Multi-Science is more abstractive since it contains more novel words
- DialogSum is more extractive since it has higher density, coverage and copy length, which means more content in the summaries can be extracted from the source text.







