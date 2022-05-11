from dataclasses import asdict
import multiprocessing
import os

from example_funcs import (  # noqa it depends on your task; you could also customized the function
    text_classification_func,
)

from datalabs import load_dataset
from datalabs.operations.aggregate.text_classification import (
    get_features_dataset_level as get_features_dataset_level_text_classification,
)
from datalabs.operations.preprocess.general import tokenize
from datalabs.tasks.text_classification import TextClassification
from datalabs.utils.more_features import get_features_dataset, prefix_dict_key

"""
pip install --upgrade pip
pip install datalabs
python -m nltk.downloader omw-1.4 # to support more feature calculation
"""


def get_info(directory_of_files, language, task):

    # add sample-level features
    features_mongodb = {}
    dataset = {}

    for file_name in sorted(os.listdir(directory_of_files)):

        if file_name not in ["train.json", "validation.json", "test.json"]:
            continue
        else:
            split_name = file_name.split(".json")[0]

            # language = "en"
            # task = "text-classification"
            path_dataset = directory_of_files + "/" + file_name

            dataset_tmp = load_dataset("json", data_files=path_dataset)
            dataset[split_name] = dataset_tmp["train"]
            dataset[split_name]._info.task_templates = [TextClassification(task)]
            dataset[split_name]._info.languages = [language]

            raw_features = asdict(dataset[split_name]._info)["features"]
            dataset[split_name] = dataset[split_name].apply(
                tokenize, num_proc=multiprocessing.cpu_count(), mode="memory"
            )
            dataset[split_name] = dataset[split_name].apply(
                text_classification_func,
                num_proc=multiprocessing.cpu_count(),
                mode="memory",
            )

            all_features = asdict(dataset[split_name]._info)["features"]

            # turn on advanced fields
            for feature_name, feature_info in all_features.items():
                # this is defined for the case when feature is `text_tokenized`
                if (
                    feature_name.find("tokenize") != -1
                    and all_features[feature_name]["dtype"] == "string"
                ):
                    feature_info["raw_feature"] = True
                    feature_info["is_bucket"] = False
                elif feature_name not in raw_features.keys():
                    feature_info["raw_feature"] = False
                    feature_info["is_bucket"] = True

            features_mongodb.update(all_features)
            # calculate dataset-level features
            dataset[split_name] = dataset[split_name].apply(
                get_features_dataset_level_text_classification,
                mode="memory",
                prefix="avg",
            )

            features_dataset = get_features_dataset(dataset[split_name]._stat)
            for attr, feat_info in features_dataset.items():
                feat_info = asdict(feat_info)
                value = dataset[split_name]._stat[attr]
                feat_info["value"] = value
                features_dataset[attr] = feat_info
            features_dataset_new = prefix_dict_key(features_dataset, prefix=split_name)

            # add dataset-level features
            features_mongodb.update(features_dataset_new)
            metadata = asdict(dataset[split_name]._info)
            metadata["features"] = features_mongodb

    return asdict(dataset["train"]._info), features_mongodb, dataset


def get_paper_template(
    year=None,
    venue=None,
    title=None,
    author=None,
    url=None,
    bib=None,
):
    return {
        "year": year,
        "venue": venue,
        "title": title,
        "author": author,
        "url": url,
        "bib": bib,
    }


def get_annotator_demographic_template(
    gender=None,
    race=None,
    native_language=None,
    socioeconomic_status=None,
    number_of_different_speakers_represented=None,
    presence_of_disordered_speech=None,
    training_in_linguistics=None,
):
    return {
        "gender": gender,
        "race_ethnicity": race,
        "native_language": native_language,
        "socioeconomic_status": socioeconomic_status,
        "number_of_different_speakers_represented": number_of_different_speakers_represented,  # noqa
        "presence_of_disordered_speech": presence_of_disordered_speech,
        "training_in_linguistics": training_in_linguistics,
    }


def get_speaker_demographic_template(
    gender=None,
    race=None,
    native_language=None,
    socioeconomic_status=None,
    number_of_different_speakers_represented=None,
    presence_of_disordered_speech=None,
):
    return {
        "gender": gender,
        "race_ethnicity": race,
        "native_language": native_language,
        "socioeconomic_status": socioeconomic_status,
        "number_of_different_speakers_represented": number_of_different_speakers_represented,  # noqa
        "presence_of_disordered_speech": presence_of_disordered_speech,
    }


def get_speech_situation_templates(
    time=None, place=None, modality=None, intended_audience=None
):
    return {
        "time": time,
        "place": place,
        "modality": modality,
        "intended_audience": intended_audience,
    }


def get_popularity_template(
    number_of_download=None,
    number_of_likes=None,
    number_of_reposts=None,
    number_of_visits=None,
):
    return {
        "number_of_download": number_of_download,
        "number_of_likes": number_of_likes,
        "number_of_reposts": number_of_reposts,
        "number_of_visits": number_of_visits,
    }


def get_size_template(samples=None, storage=None):
    return {
        "sample": samples,
        "storage": storage,
    }


def get_transformation_template(type):
    return {"type": type}


def validate_generate_db_metadata(
    dataset_name,
    transformation,
    version,
    task_categories,
    tasks,
    split,
    languages,
    sub_dataset=None,
    summary=None,
    homepage=None,
    repository=None,
    leaderboard=None,
    person_of_contact=None,
    features=None,
    speaker_demographic=None,
    annotator_demographic=None,
    speech_situation=None,
    size=None,
    license_=None,
    huggingface_link=None,
    curation_rationale=None,
    genre=None,
    quality=None,
    similar_datasets=None,
    popularity=None,
    creator_name=None,
    multilinguality=None,
    paper_info=None,
    prompt_infos=None,
    submitter_name=None,
    system_metadata_ids=None,
    data_typology=None,
    statistical_information=None,
):
    if dataset_name is None or dataset_name == "":
        raise Exception("dataset_name should not be None or ''")
    if transformation is None:
        raise Exception("transformation should not be None")
    if version is None or version == "":
        raise Exception("version should not be None or ''")

    if tasks is None or (not isinstance(tasks, list)) or len(tasks) == 0:
        raise Exception(
            "tasks should not be None and this object should be list type"
            " and its length should not be zero"
        )
    if (
        task_categories is None
        or (not isinstance(task_categories, list))
        or len(task_categories) == 0
    ):
        raise Exception(
            "task_categories should not be None and this object should be "
            "list type and its length should not be zero"
        )

    if split is None or (not isinstance(split, dict)):
        raise Exception("split should not be None and this object should be dict type")

    if languages is None or (not isinstance(languages, list)) or len(languages) == 0:
        raise Exception(
            "languages should not be None and this object should be list"
            " type and its length should not be zero"
        )

    return {
        "dataset_name": dataset_name,
        "sub_dataset": sub_dataset,
        "split": split,
        "summary": summary,
        "homepage": homepage,
        "repository": repository,
        "paper_info": paper_info,
        "leaderboard": leaderboard,
        "person_of_contact": person_of_contact,
        "tasks": tasks,
        "task_categories": task_categories,
        "language": languages,
        "features": features,
        "speaker_demographic": speaker_demographic,
        "annotator_demographic": annotator_demographic,
        "speech_situation": speech_situation,
        "size": size,
        "license": license_,
        "huggingface_link": huggingface_link,
        "curation_rationale": curation_rationale,
        "genre": genre,
        "quality": quality,
        "similar_datasets": similar_datasets,
        "creator_name": creator_name,
        "submitter_name": submitter_name,
        "multilinguality": multilinguality,
        "system_metadata_ids": system_metadata_ids,
        "popularity": popularity,
        "transformation": transformation,
        "version": version,
        "prompt_infos": prompt_infos,
        "data_typology": data_typology,
        "statistical_information": statistical_information,
        "source": "user",
    }


def generate_db_metadata_from_sdk(
    metadata,
    features,
    dataset_name_db,
    transformation=get_transformation_template("origin"),  # noqa
    version="0.0.1",
    languages=["en"],
    data_typology="textdataset",
):

    summary = metadata["description"]
    homepage = metadata["homepage"]
    license = metadata["license"]
    languages = metadata["languages"]
    subset_name = metadata["config_name"]
    repository = metadata["repository"]
    leaderboard = metadata["leaderboard"]
    person_of_contact = metadata["person_of_contact"]
    huggingface_link = metadata["huggingface_link"]
    curation_rationale = metadata["curation_rationale"]
    genre = metadata["genre"]
    similar_datasets = metadata["similar_datasets"]
    creator_id = metadata["creator_id"]
    submitter_id = metadata["submitter_id"]
    multilinguality = metadata["multilinguality"]
    speaker_demographic = metadata["speaker_demographic"]
    annotator_demographic = metadata["annotator_demographic"]
    speech_situation = metadata["speech_situation"]
    prompt_infos = metadata["prompts"]

    task_categories = []  #
    tasks = []  #
    for value in metadata["task_templates"]:
        task_categories.append(value["task_category"])
        tasks.append(value["task"])

    split = {}  #
    for key in metadata["splits"].keys():
        split[key] = metadata["splits"][key]["num_examples"]

    return validate_generate_db_metadata(
        dataset_name=dataset_name_db,
        transformation=transformation,
        version=version,
        task_categories=task_categories,
        tasks=tasks,
        split=split,
        languages=languages,
        summary=summary,
        homepage=homepage,
        license_=license,
        sub_dataset=subset_name,
        repository=repository,
        leaderboard=leaderboard,
        person_of_contact=person_of_contact,
        huggingface_link=huggingface_link,
        curation_rationale=curation_rationale,
        genre=genre,
        similar_datasets=similar_datasets,
        creator_name=creator_id,
        submitter_name=submitter_id,
        multilinguality=multilinguality,
        speaker_demographic=speaker_demographic,
        annotator_demographic=annotator_demographic,
        speech_situation=speech_situation,
        features=features,
        data_typology=data_typology,
        prompt_infos=prompt_infos,
    )


def client(
    dataset_name_db,
    directory_of_files,
    language,
    task,
    transformation={"type": "origin"},
    version="origin",
    data_typology="textdataset",
):

    metadata_sdk, metadata_features_sdk, dataset_sdk = get_info(
        directory_of_files, language, task
    )

    print(dataset_sdk)
    print(metadata_sdk.keys())

    # reformat the metadata information for db
    metadata_db = generate_db_metadata_from_sdk(
        metadata=metadata_sdk,
        features=metadata_features_sdk,
        dataset_name_db=dataset_name_db,
        transformation=transformation,
        version=version,
        languages=language,
        data_typology=data_typology,
    )

    MAX_NUMBER_OF_SAMPLES = 100000
    samples_db = []
    for split in dataset_sdk.keys():
        for idx, sample in enumerate(dataset_sdk[split]):
            if idx > MAX_NUMBER_OF_SAMPLES:
                break
            samples_db.append({"split_name": split, "features": sample})

    print(metadata_db, samples_db)

    """
    ## call the API
    data_json = {
    'metadata': metadata_db,
    'samples': samples_db,
    'user_name': self.user_name,
    'password': self.password,
    'role': self.role,
    'status': self.status,
    }
    """


# ----------- Example -----------------
# directory_of_files: the path of user-uploaded data
dataset_name_db = "qc"
directory_of_files = "./" + dataset_name_db  # ./qc
language = "en"
task = "text-classification"
transformation = {"type": "origin"}
version = "origin"
data_typology = "textdataset"

client(
    dataset_name_db,
    directory_of_files,
    language,
    task,
    transformation,
    version,
    data_typology,
)
