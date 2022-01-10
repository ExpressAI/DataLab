## MEMO

DatasetInfo 类和 MongoDB `metadata` 字段的区别（以 `ag_news` 为例）：
1. 没有保存在 MongoDB 中的字段：`download_checksums`、`download_size`、`post_processing_size`、`dataset_size`、`size_in_bytes`、`builder_name`、`config_name`、`post_processed`、`supervised_keys`、`homepage`
2. 新增加的字段：

    ```json
    {
        "_id": {
            "$oid": "617685b833e51a7edda1cecb"
        },
        "dataset_name": "ag_news",
        "sub_dataset": null,
        "homepage": null,
        "repository": null,
        "leaderboard": "https://paperswithcode.com/dataset/ag-news",
        "person_of_contact": null,
        "languages": [
            "en"
        ],
        "features": {
            "text_length": {},
            "text_basic_words": {},
            "text_lexical_richness": {},
            "text_flesch_reading_ease": {},
            "text_gender_bias_word_male": {},
            "text_gender_bias_word_female": {},
            "text_gender_bias_single_name_male": {},
            "text_gender_bias_single_name_female": {},
            "text_gender_bias_name_male": {},
            "text_gender_bias_name_female": {},
            "text_hate_speech": {},
            "text_train_avg_length": {},
            "text_train_avg_basic_words": {},
            "text_train_avg_lexical_richness": {},
            "text_train_avg_gender_bias_word_male": {},
            "text_train_avg_gender_bias_word_female": {},
            "text_train_avg_gender_bias_single_name_male": {},
            "text_train_avg_gender_bias_single_name_female": {},
            "text_train_avg_gender_bias_name_male": {},
            "text_train_avg_gender_bias_name_female": {},
            "text_test_avg_length": {},
            "text_test_avg_basic_words": {},
            "text_test_avg_lexical_richness": {},
            "text_test_avg_gender_bias_word_male": {},
            "text_test_avg_gender_bias_word_female": {},
            "text_test_avg_gender_bias_single_name_male": {},
            "text_test_avg_gender_bias_single_name_female": {},
            "text_test_avg_gender_bias_name_male": {},
            "text_test_avg_gender_bias_name_female": {}
        },
        "speaker_demographic": {
            "gender": null,
            "race": null,
            "ethnicity": null,
            "native language": null,
            "socioeconomic status": null,
            "number_of_different_speakers_represented": null,
            "presence_of_disordered_speech": null,
            "training_in_linguistics": null
        },
        "annotator_demographic": {
            "gender": null,
            "race": null,
            "ethnicity": null,
            "native language": null,
            "socioeconomic status": null,
            "number_of_different_speakers_represented": null,
            "presence_of_disordered_speech": null,
            "training_in_linguistics": null
        },
        "speech_situation": {
            "time": null,
            "place": null,
            "modality": null,
            "intended_audience": null
        },
        "size": {
            "samples": {
                "$numberInt": "127600"
            },
            "storage": "30.2284517288208 mega bytes"
        },
        "production_status": null,
        "huggingface_link": "https://huggingface.co/datalab/ag_news",
        "curation_rationale": null,
        "genre": null,
        "quality": null,
        "similar_datasets": null,
        "creator_id": null,
        "submitter_id": null,
        "Multilinguality": null,
        "model_ids": [],
        "popularity": {
            "number_of_download": {
                "$numberInt": "0"
            },
            "number_of_times": {
                "$numberInt": "0"
            },
            "number_of_reposts": {
                "$numberInt": "0"
            },
            "number_of_visits": {
                "$numberInt": "0"
            }
        },
        "transformation": "origin"
    }
    ```
    
3. 疑似有对应的字段（每个块上方来自 DatasetInfo，下方来自 MongoDB 的 `metadata`）
    1. `version`:
    
        ```json
        "version": {
            "version_str": "0.0.0",
            "description": null,
            "major": 0,
            "minor": 0,
            "patch": 0
        }
        
        "version": null
        ```
    
    2. `task_templates`
    
        ```json
        "task_templates": [
            {
            "task": "text-classification",
            "text_column": "text",
            "label_column": "label",
            "labels": [
                "Business",
                "Sci/Tech",
                "Sports",
                "World"
            ]
            }
        ]
        
        "task": [
            "topic-classification"
        ],
        "task_category": [
            "text-classification"
        ],
        ```
    
    3. 引用：
    
        ```json
        "citation": "@inproceedings{Zhang2015CharacterlevelCN,\n  title={Character-level Convolutional Networks for Text Classification},\n  author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},\n  booktitle={NIPS},\n  year={2015}\n}\n"
        
        "paper": {
            "year": null,
            "venue": null,
            "title": null,
            "author": null,
            "url": null,
            "bib": null,
            "citation": {
            "2018": {
                "$numberInt": "0"
            },
            "2019": {
                "$numberInt": "0"
            },
            "2020": {
                "$numberInt": "0"
            },
            "2021": {
                "$numberInt": "0"
            },
            "2022": {
                "$numberInt": "0"
            },
            "all": {
                "$numberInt": "0"
            }
            }
        },
        ```
    

<p align="right"> 12/09 </p>
