# %%
import json
import pickle

import fire
import pandas as pd

from datalabs import load_dataset, Prompt


def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def process_prompt(dataset, signal_type, outpath):
    """
    dataset: the name of the dataset we want to process prompts
    signal_type: a list of strings, separated by comma
    outpath: the path to the output prompts file
    """
    df = pd.read_pickle("datasets/promptsource.pkl")
    tgts = []
    for row in df.itertuples():
        if row.dataset == dataset:
            tgts.append(row)

    outfile = open(outpath, "w")
    prompts = []
    count = 0
    for tgt in tgts:
        if not tgt.original_task:
            continue
        # if answer_choices is not None
        template = tgt.jinja
        if tgt.answer_choices is not None:
            # Only classification has answers
            template = template.replace("answer_choices", "answers")
            answers = tgt.answer_choices
            answers = answers.split("|||")
            answers = [a.strip() for a in answers]

            datalab_ds = load_dataset(dataset)
            datalab_labels = (
                datalab_ds["train"]._info.__dict__["features"]["label"].names
            )
            assert len(datalab_labels) == len(answers)
            answers = [
                {datalab_label: [answer]}
                for datalab_label, answer in zip(datalab_labels, answers)
            ]
            # answers = {idx: label for idx, label in enumerate(answers)}
        else:
            answers = None
        features = {
            "length": len(template.split(" ")),
            "shape": "prefix",
            "skeleton": tgt.name,
        }
        reference = tgt.reference
        if reference is None or len(reference) == 0:
            reference = "https://arxiv.org/abs/2202.01279"
        prompt = Prompt(
            id=str(count),
            language="en",
            description="We use ||| to separate source and target in a template.",
            template=template,
            answers=answers,
            supported_plm_types=["left-to-right", "encoder-decoder"],
            signal_type=signal_type.split(","),
            features=features,
            reference=reference,
        )
        count += 1
        prompts.append(prompt.__dict__)
    print(json.dumps(prompts, indent=4), file=outfile)
    outfile.flush()


if __name__ == "__main__":
    fire.Fire(process_prompt)

# python generate_prompt_sketch.py --dataset ag_news --signal_type text-classification --outpath datasets/ag_news/prompts.json
