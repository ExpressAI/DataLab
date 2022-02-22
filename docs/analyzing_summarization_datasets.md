# Analyzing Summarization Datasets


## Supported Datasets

|Datasets|Version|Task Schema|Dataloader|Comments
|:---    |:---        |:---       |:--- |:---  
govreport | -  | Summarization | ``` load_dataset("govreport") ``` | Current definition: text, summary |
dialogsum | `document`  | Summarization | ``` load_dataset("dialogsum", "document") ``` | Current definition: text, summary |
dialogsum | `dialogue`  | Summarization | ``` load_dataset("dialogsum", "dialogue") ``` | Current definition: dialogue: {"speaker": List[str], "text": List[str]}, summary: List[str] |
