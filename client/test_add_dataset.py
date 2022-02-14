import requests
import json
if __name__ == "__main__":
    metadata = {
        "dataset_name": "ag_news_test_yt",
        "sub_dataset": 'null',
        "split": {"train": {"$numberInt": "120000"}, "test": {"$numberInt": "7600"}},
        "summary": "AG is a collection of more than 1 million news articles. News articles have been\ngathered from more than 2000 news sources by ComeToMyHead in more than 1 year of\nactivity. ComeToMyHead is an academic news search engine which has been running\nsince July, 2004. The dataset is provided by the academic comunity for research\npurposes in data mining (clustering, classification, etc), information retrieval\n(ranking, search, etc), xml, data compression, data streaming, and any other\nnon-commercial activity. For more information, please refer to the link\nhttp://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .\n\nThe AG's news topic classification dataset is constructed by Xiang Zhang\n(xiang.zhang@nyu.edu) from the dataset above. It is used as a text\nclassification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann\nLeCun. Character-level Convolutional Networks for Text Classification. Advances\nin Neural Information Processing Systems 28 (NIPS 2015).\n",
        "homepage": "http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
        "repository": None,
        "leaderboard": "https://paperswithcode.com/dataset/ag-news", "person_of_contact": None,
        "features": {
            "text": {"type": "string", "label": False, "raw_feature": True, "mapping": None, "dataset_level": None,
                     "sample_level": None},
            "label": {"type": "int", "label": True, "raw_feature": True,
                      "mapping": ["World", "Sports", "Business", "Sci/Tech"], "dataset_level": None,
                      "sample_level": None}, },
        "speaker_demographic": {"gender": None, "native language": None, "socioeconomic status": None,
                                "number_of_different_speakers_represented": None, "presence_of_disordered_speech": None,
                                "race_ethnicity": None},
        "annotator_demographic": {"gender": None, "native language": None, "socioeconomic status": None,
                                  "number_of_different_speakers_represented": None,
                                  "presence_of_disordered_speech": None, "training_in_linguistics": None,
                                  "race_ethnicity": None},
        "speech_situation": {"time": None, "place": None, "modality": None, "intended_audience": None},
        "size": {"sample": {"$numberInt": "127600"},
                 "storage": ""},
        "license": None,
        "huggingface_link": "https://huggingface.co/datasets/ag_news",
        "curation_rationale": None, "genre": None, "quality": None,
        "similar_datasets": "[\"pragmeval\", \"offenseval2020_tr\", \"hate_offensive\", \"hda_nli_hindi\", \"ar_res_reviews\"]",
        "popularity": {"number_of_download": {"$numberInt": "10415"}, "number_of_times": {"$numberInt": "0"},
                       "number_of_reposts": {"$numberInt": "10936"}, "number_of_visits": {"$numberInt": "10794"},
                       "number_of_likes": {"$numberInt": "10106"}},
        "transformation": {"type": "origin"},
        "version": "Hugging Face",
        "creator_name": None,
        "language": ["en"],
        "multilinguality": None,
        "paper_info": {"year": "2015", "venue": "NIPS",
                       "title": "Character-level Convolutional Networks for Text Classification",
                       "author": "\"Xiang Zhang, Junbo Zhao, Yann LeCun\"",
                       "url": "https://paperswithcode.com/paper/character-level-convolutional-networks-for",
                       "bib": "https://dblp.org/rec/conf/nips/ZhangZL15.html?view=bibtex"}, "prompt_infos": [],
        "submitter_name": None,
        "system_metadata_ids": [],
        "task_categories": ["text-classification"],
        "tasks": ["topic-classification"],
        "data_typology": "textdataset",
        "pwc_description": "",
        "introducing_paper_title": ""}
    samples = [
        {"split_name": "train",
         "features": {
             "text": "Oil prices soar to all-time record, posing new menace to US economy (AFP) AFP - Tearaway world oil prices, toppling records and straining wallets, present a new economic menace barely three months before the US presidential elections.",
             "label": {"$numberInt": "2"}, "text_length": {"$numberInt": "37"},
         }, }
    ]
    data = {
        'metadata': metadata,
        'samples': samples
    }
    path = 'http://3.23.213.76:5001/upload_new_dataset'
    r = requests.post(path, json=data)
    print(json.dumps(data))
    print(r)
