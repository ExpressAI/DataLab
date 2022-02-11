import json
import sys
import warnings
from collections import defaultdict
from time import gmtime, strftime
from typing import List, Dict

import requests
from tqdm import trange




end_point_upload_dataset = "http://3.23.213.78:5001/upload_new_dataset"

dataset_name = "test_by_pengfei"


data_info = {
  "metadata": {
    "dataset_name": dataset_name,
    "sub_dataset": "string",
    "split": {
      "train": 1000,
      "test": 100
    },
    "summary": "string",
    "homepage": "string",
    "repository": "string",
    "leaderboard": "string",
    "person_of_contact": "string",
    "features": {},
    "speaker_demographic": {
      "gender": "string",
      "native language": "string",
      "socioeconomic status": "string",
      "number_of_different_speakers_represented": "string",
      "presence_of_disordered_speech": "string",
      "race_ethnicity": "string"
    },
    "annotator_demographic": {
      "gender": "string",
      "native language": "string",
      "socioeconomic status": "string",
      "number_of_different_speakers_represented": "string",
      "presence_of_disordered_speech": "string",
      "race_ethnicity": "string",
      "training_in_linguistics": "string"
    },
    "speech_situation": {
      "time": "2022-02-10T15:49:52.127Z",
      "place": "string",
      "modality": "string",
      "intended_audience": "string"
    },
    "size": {
      "sample": 5000,
      "storage": "50MB"
    },
    "license": "string",
    "huggingface_link": "string",
    "curation_rationale": {},
    "genre": {},
    "transformation": {
      "type": "string"
    },
    "version": "string",
    "creator_name": "string",
    "language": [
      "string"
    ],
    "multilinguality": "string",
    "paper_info": {
      "year": "2022-02-10",
      "venue": "string",
      "title": "string",
      "author": "string",
      "url": "string",
      "bib": "string"
    },
    "submitter_name": "string",
    "task_categories": [
      "text-classification"
    ],
    "tasks": [
      "sentiment-classification"
    ],
    "data_typology": "textdataset"
  },
  "samples": [
    {
      "split_name": "train",
      "features": {}
    }
  ]
}



response = requests.post(url=end_point_upload_dataset, json=json.dumps(data_info))

print(response)