# Unify tasks


## [Text Classification](https://github.com/ExpressAI/DataLab/blob/1dc063aabe6d245b0cfd323780ecba563fd5e630/src/datalabs/tasks/text_classification.py#L24)

This is the schema for text classification task.
```json
{
    "text_column":"string",
    "label_column":"string", 
}
```

#### dbpedia_14

* dataset_name: dbpedia_14
* sub_dataset: dbpedia_14
* schema instantiation
    * text_column: content
    * label_column: label

  
| Dataset name           | Subset                     | Text Column         | Label Column                                                                                               |
|------------------------|----------------------------|---------------------|------------------------------------------------------------------------------------------------------------|
| allocine               | None                       | text                | label                                                                                                      |
| ethos                  | binary                     | text                | label                                                                                                      |
| ethos                  | multilabel                 | text                | violence, directed_vs_generalized, gender, race, national_origin, disability, religion, sexual_orientation |
| eurlex                 | None                       | text                | eurovoc_concepts                                                                                           |
| gnad10                 | None                       | text                | label                                                                                                      |
| pragmeval              | verifiability              | text                | label                                                                                                      |
| pragmeval              | persuasiveness-eloquence   | text1, text2        | label                                                                                                      |
| pragmeval              | emobank-arousal            | text                | label                                                                                                      |
| pragmeval              | switchboard                | text                | label                                                                                                      |
| pragmeval              | mrda                       | text                | label                                                                                                      |
| pragmeval              | gum                        | text1, text2        | label                                                                                                      |
| pragmeval              | emergent                   | text1, text2        | label                                                                                                      |
| pragmeval              | persuasiveness-relevance   | text1, text2        | label                                                                                                      |
| pragmeval              | persuasiveness-specificity | text1, text2        | label                                                                                                      |
| pragmeval              | persuasiveness-strength    | text1, text2        | label                                                                                                      |
| pragmeval              | emobank-dominance          | text                | label                                                                                                      |
| pragmeval              | squinky-implicature        | text                | label                                                                                                      |
| pragmeval              | sarcasm                    | text1, text2        | label                                                                                                      |
| pragmeval              | squinky-formality          | text                | label                                                                                                      |
| pragmeval              | stac                       | text1, text2        | label                                                                                                      |
| pragmeval              | pdtb                       | text1, text2        | label                                                                                                      |
| pragmeval              | persuasiveness-premisetype | text1, text2        | label                                                                                                      |
| pragmeval              | squinky-informativeness    | text                | label                                                                                                      |
| pragmeval              | persuasiveness-claimtype   | text1, text2        | label                                                                                                      |
| pragmeval              | emobank-valence            | text                | label                                                                                                      |
| offenseval2020_tr      | None                       | tweet               | subtask_a                                                                                                  |
| hate_offensive         | None                       | tweet               | label                                                                                                      |
| hda_nli_hindi          | None                       | premise, hypothesis | label                                                                                                      |
| hatexplain             | None                       |                     |                                                                                                            |
| ar_res_reviews         | None                       | text                | polarity                                                                                                   |
| catalonia_independence | catalan                    | TWEET               | LABEL                                                                                                      |
| catalonia_independence | spanish                    | TWEET               | LABEL                                                                                                      |
| bbc_hindi_nli          | None                       | premise, hypothesis | label                                                                                                      |
| cedr                   | main                       | text                | labels                                                                                                     |
| cedr                   | enriched                   | text                | labels                                                                                                     |
| arsentd_lev            | None                       | Tweet               | Sentiment                                                                                                  |


To be continued...



