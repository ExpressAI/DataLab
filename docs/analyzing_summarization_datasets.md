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


