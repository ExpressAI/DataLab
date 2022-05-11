from typing import Any, Callable, Dict, List, Mapping, Optional

from datalabs.operations.operation import dataset_operation, DatasetOperation
from datalabs.operations.prompt.prompting import Prompting, prompting


class SentimentClassificationPrompting(Prompting, DatasetOperation):
    def __init__(
        self,
        name: str = None,
        func: Callable[..., Any] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text", "label"],
        generated_field: str = None,
        task="sentiment-classification",
        description=None,
        template=None,
    ):
        super().__init__(
            name=name,
            func=func,
            resources=resources,
            contributor=contributor,
            task=task,
            description=description,
        )
        self._type = "SentimentClassificationPrompting"
        self.processed_fields = ["text", "label"]
        if isinstance(processed_fields, str):
            self.processed_fields[0] = processed_fields
        else:
            self.processed_fields = processed_fields
        self.generated_field = generated_field
        self._data_type = "Dataset"
        self.template = template

    def __call__(self, sample, labels_to_answers) -> Any:  # str?
        """
        Parameters
        x: Text

        Returns
        Transformed Text
        """
        return self.func(sample, labels_to_answers, **self.resources)


class sentiment_classification_prompting(prompting, dataset_operation):
    def __init__(
        self,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        contributor: str = None,
        processed_fields: List = ["text", "label"],
        generated_field: str = None,
        task="sentiment-classification",
        description=None,
        template=None,
    ):
        super().__init__(
            name=name,
            resources=resources,
            contributor=contributor,
            description=description,
        )
        self.processed_fields = processed_fields
        self.generated_field = generated_field
        self.task = task
        self.template = template

    def __call__(self, *param_arg):
        if callable(self.name):
            tf_class = SentimentClassificationPrompting(
                name=self.name.__name__, func=self.name
            )
            return tf_class(*param_arg)
        else:
            f = param_arg[0]
            name = self.name or f.__name__
            tf_cls = SentimentClassificationPrompting(
                name=name,
                func=f,
                resources=self.resources,
                contributor=self.contributor,
                processed_fields=self.processed_fields,
                generated_field=self.generated_field,
                task=self.task,
                description=self.description,
                template=self.template,
            )
            return tf_cls


"""
>>> labels = [positive, negative]
>>> text = "I love this movie"
>>> texture_choice = ", ".join(labels[:-1]) + " or " + labels[-1] + "?"
>>> template = f"Given the text {text}, is it {texture_choice}"
>>> template
'Given the text I love this movie, is it positive or negative?';


Test Example:

from datalabs import load_dataset
dataset = load_dataset('mr')
from prompt.sentiment_classification import *
res = dataset['test'].apply(template_p1)
print(next(res))



"""


@sentiment_classification_prompting(
    name="template_sc1",
    contributor="datalab",
    template="Given the text: {text}, is it {texture_choices}",
    description="Prompt template: Given the text: "
    "{text}, is it"
    " {texture_choices}",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc1(sample: dict, labels_to_answers: Dict):
    tp1 = "Given the text: {text}, is it {texture_choices}"

    # prompting process
    answers = list(labels_to_answers.values())
    text = sample["text"]  # noqa
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp1))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@sentiment_classification_prompting(
    name="template_sc2",
    contributor="datalab",
    template="Given the text: {text}, it is [mask]",
    description="Prompt template: Given the text: {text}, it is [mask]",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc2(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text}, it is [mask]"

    # prompting process
    text = sample["text"]  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@sentiment_classification_prompting(
    name="template_sc3",
    contributor="datalab",
    template="Given the text: {text} Judge the sentiment of this text."
    " You may choose from {texture_choices}.",
    description="Prompt template: Given the text: {text} Judge the sentiment"
    " of this text. You may choose from {texture_choices}.",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc3(sample: dict, labels_to_answers: Dict):
    tp = (
        "Given the text: {text} Judge the sentiment of this text. "
        "You may choose from {texture_choices}."
    )

    # prompting process
    text = sample["text"]  # noqa
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers)  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@sentiment_classification_prompting(
    name="template_sc4",
    contributor="datalab",
    template="Given the text: {text} What's the sentiment of this"
    " text? {texture_choices}",
    description="Prompt template: Given the text: {text} What's the"
    " sentiment of this text? {texture_choices}",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc4(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} What\\'s the sentiment of this text? {texture_choices}"

    # prompting process
    text = sample["text"]  # noqa
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@sentiment_classification_prompting(
    name="template_sc5",
    contributor="datalab",
    template="Given the text: {text} Can you tell the sentiment of the"
    " text? {texture_choices}",
    description="Prompt template: Given the text: {text} Can you tell "
    "the sentiment of the text? {texture_choices}",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc5(sample: dict, labels_to_answers: Dict):
    tp = (
        "Given the text: {text} Can you tell the sentiment"
        " of the text? {texture_choices}"
    )

    # prompting process
    text = sample["text"]  # noqa
    answers = list(labels_to_answers.values())
    texture_choices = ", ".join(answers[:-1]) + " or " + answers[-1] + "?"  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}


@sentiment_classification_prompting(
    name="template_sc6",
    contributor="datalab",
    template="Given the text: {text} The sentiment of the text is [mask]",
    description="Prompt template: Prompt template: Given the text: {text} "
    "The sentiment of the text is [mask]",
    task="sentiment-classification",
    processed_fields=["text", "label"],
)
def template_sc6(sample: dict, labels_to_answers: Dict):
    tp = "Given the text: {text} The sentiment of the text is [mask]"

    # prompting process
    text = sample["text"]  # noqa

    # instantiation
    text_prompt = eval("f'{}'".format(tp))

    label_prompt = labels_to_answers[sample["label"]]

    return {"text_prompt": text_prompt, "label_prompt": label_prompt}
