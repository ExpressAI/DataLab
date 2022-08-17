# coding=utf-8
"""RST signals"""

import ast
import csv
import json
import os
import textwrap
from typing import List

import datalabs
from datalabs import get_task, TaskType
from datalabs.features import Features, Sequence, Value
from datalabs.tasks.question_answering import QuestionAnsweringMultipleChoice

_RST_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2206.11147,
  doi = {10.48550/ARXIV.2206.11147},
  url = {https://arxiv.org/abs/2206.11147},
  author = {Yuan, Weizhe and Liu, Pengfei},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {reStructured Pre-training},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_RST_DESCRIPTION = """\
RST is a collection of various kinds of signals naturally exist in the world.
"""


class RSTConfig(datalabs.BuilderConfig):
    def __init__(
            self,
            data_url,
            data_dir,
            citation,
            url,
            features,
            process_label=lambda x: x,
            task_templates=None,
            **kwargs
    ):
        """BuilderConfig for RST.

        Args:
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(RSTConfig, self).__init__(version=datalabs.Version("1.0.0", ""), **kwargs)

        self.features = features
        self.data_url = data_url
        self.citation = citation
        self.url = url
        self.process_label = process_label
        self.task_templates = task_templates


class RST(datalabs.GeneratorBasedBuilder):
    """ RST signals """
    BUILDER_CONFIGS = [
        # Rotten Tomatoes
        RSTConfig(
            name="rotten_tomatoes_sentiment",
            description=textwrap.dedent(
                """\
            Sentiment signals from Rotten Tomatoes"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/rotten_tomatoes_sentiment_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "review": Value("string"),
                    "rating": Value("float")
                }
            )
        ),

        # Daily Mail
        RSTConfig(
            name="daily_mail_category",
            description=textwrap.dedent(
                """\
            Category signals from Daily Mail"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/daily_mail_category_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "category": Value("string")
                }
            )
        ),
        RSTConfig(
            name="daily_mail_summary",
            description=textwrap.dedent(
                """\
            Summary signals from Daily Mail"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/daily_mail_summary_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "title": Value("string"),
                    "text": Value("string"),
                    "summary": Value("string")
                }
            )
        ),
        RSTConfig(
            name="daily_mail_temporal",
            description=textwrap.dedent(
                """\
            Summary signals from Daily Mail"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/daily_mail_temporal_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "events": Sequence(Value("string")),
                }
            )
        ),

        # Wikidata
        RSTConfig(
            name="wikidata_entity",
            description=textwrap.dedent(
                """\
            Summary signals from Wikidata"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikidata_entity_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "entity": Value("string"),
                    "entity_type": Value("string"),
                    "text": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wikidata_relation",
            description=textwrap.dedent(
                """\
            Summary signals from Wikidata"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikidata_relation_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "subject": Value("string"),
                    "object": Value("string"),
                    "relation": Value("string"),
                    "text": Value("string"),
                }
            )
        ),

        # wikiHow
        RSTConfig(
            name="wikihow_text_category",
            description=textwrap.dedent(
                """\
            Text category signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_text_category_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "category": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wikihow_category_hierarchy",
            description=textwrap.dedent(
                """\
            Category hierarchy signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_category_hierarchy_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "low_category": Value("string"),
                    "high_category": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wikihow_goal_step",
            description=textwrap.dedent(
                """\
            Goal-step signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_goal_step_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "goal": Value("string"),
                    "steps": Sequence(Value("string")),
                }
            )
        ),
        RSTConfig(
            name="wikihow_summary",
            description=textwrap.dedent(
                """\
            Summary signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_summary_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "summary": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wikihow_procedure",
            description=textwrap.dedent(
                """\
            Procedure signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_procedure_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "goal": Value("string"),
                    "first_step": Value("string"),
                    "second_step": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wikihow_question",
            description=textwrap.dedent(
                """\
            Question signals from wikiHow"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikihow_question_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "question": Value("string"),
                    "description": Value("string"),
                    "answer": Value("string"),
                    "related_questions": Sequence(Value("string")),
                }
            )
        ),

        # Wikipedia
        RSTConfig(
            name="wikipedia_entities",
            description=textwrap.dedent(
                """\
            Entity signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikipedia_entities_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "entities": Sequence(Value("string")),
                }
            )
        ),
        RSTConfig(
            name="wikipedia_sections",
            description=textwrap.dedent(
                """\
            Section title signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wikipedia_sections_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "texts": Sequence(Value("string")),
                    "titles": Sequence(Value("string")),
                }
            )
        ),

        # WordNet
        RSTConfig(
            name="wordnet_pos",
            description=textwrap.dedent(
                """\
            Part-of-speech signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wordnet_pos_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "word": Value("string"),
                    "sentence": Value("string"),
                    "pos": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="wordnet_meaning",
            description=textwrap.dedent(
                """\
            Meaning signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wordnet_meaning_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "word": Value("string"),
                    "sentence": Value("string"),
                    "meaning": Value("string"),
                    "possible_meanings": Sequence(Value("string")),
                }
            )
        ),
        RSTConfig(
            name="wordnet_synonym",
            description=textwrap.dedent(
                """\
            Synonym signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wordnet_synonym_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "word": Value("string"),
                    "sentence": Value("string"),
                    "synonyms": Sequence(Value("string")),
                }
            )
        ),
        RSTConfig(
            name="wordnet_antonym",
            description=textwrap.dedent(
                """\
            Antonym signals from Wikipedia"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/wordnet_antonym_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "word": Value("string"),
                    "sentence": Value("string"),
                    "antonyms": Sequence(Value("string")),
                }
            )
        ),

        # QA
        RSTConfig(
            name="qa_control",
            description=textwrap.dedent(
                """\
            QA signals from ConTRoL"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_control_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "premise": Value("string"),
                    "hypothesis": Value("string"),
                    "label": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_dream",
            description=textwrap.dedent(
                """\
            QA signals from DREAM"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_dream_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "options": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_logiqa",
            description=textwrap.dedent(
                """\
            QA signals from LogiQA"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_logiqa_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "options": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_reclor",
            description=textwrap.dedent(
                """\
            QA signals from ReClor"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_reclor_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "options": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_race",
            description=textwrap.dedent(
                """\
            QA signals from RACE"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_race_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "options": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_race_c",
            description=textwrap.dedent(
                """\
            QA signals from RACE-C"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_race_c_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "options": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="qa_triviaqa",
            description=textwrap.dedent(
                """\
            QA signals from TriviaQA"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/qa_triviaqa_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "context": Value("string"),
                    "question": Value("string"),
                    "answer": Value("string"),
                }
            )
        ),

        # arXiv
        RSTConfig(
            name="arxiv_category",
            description=textwrap.dedent(
                """\
            Category signals from arXiv"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/arxiv_category_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "category": Value("string"),
                }
            )
        ),
        RSTConfig(
            name="arxiv_summary",
            description=textwrap.dedent(
                """\
            Summary signals from arXiv"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/arxiv_summary_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "summary": Value("string"),
                }
            )
        ),

        # Paperswithcode
        RSTConfig(
            name="paperswithcode_entity",
            description=textwrap.dedent(
                """\
            Entity signals from Paperswithcode"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/paperswithcode_entity_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "entities": Sequence(Value("string")),
                    "datasets": Sequence(Value("string")),
                    "methods": Sequence(Value("string")),
                    "tasks": Sequence(Value("string")),
                    "metrics": Sequence(Value("string")),
                }
            )
        ),
        RSTConfig(
            name="paperswithcode_summary",
            description=textwrap.dedent(
                """\
            Summary signals from Paperswithcode"""
            ),
            data_url="https://storage.googleapis.com/rst-experiments/rst-data-mini/paperswithcode_summary_mini.jsonl",
            data_dir=None,
            citation=textwrap.dedent(
                """\
                TBC
            }"""
            ),
            url="TBC",
            features=datalabs.Features(
                {
                    "text": Value("string"),
                    "summary": Value("string"),
                }
            )
        ),
    ]

    def _info(self):
        return datalabs.DatasetInfo(
            description=_RST_DESCRIPTION,
            features=self.config.features,
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _RST_CITATION,
            task_templates=self.config.task_templates,
        )

    def _split_generators(self, dl_manager):
        data_file = dl_manager.download(self.config.data_url)
        return [
            datalabs.SplitGenerator(
                name=datalabs.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_file,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        if self.config.name == "rotten_tomatoes_sentiment":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    review, rating = data["review"], data["rating"]
                    yield id_, {
                        "review": review,
                        "rating": rating
                    }
        elif self.config.name == "daily_mail_category":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, category = data["text"], data["category"]
                    yield id_, {
                        "text": text,
                        "category": category
                    }

        elif self.config.name == "daily_mail_summary":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    title, text, summary = data["title"], data["text"], data["summary"]
                    yield id_, {
                        "title": title,
                        "text": text,
                        "summary": summary
                    }

        elif self.config.name == "daily_mail_temporal":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, events = data["text"], data["events"]
                    yield id_, {
                        "text": text,
                        "events": events
                    }

        elif self.config.name == "wikidata_entity":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    entity, entity_type, text = data["entity"], data["entity_type"], data["text"]
                    yield id_, {
                        "entity": entity,
                        "entity_type": entity_type,
                        "text": text
                    }

        elif self.config.name == "wikidata_relation":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    subject, object, relation, text = data["subject"], data["object"], data["relation"], data["text"]
                    yield id_, {
                        "subject": subject,
                        "object": object,
                        "relation": relation,
                        "text": text
                    }

        elif self.config.name == "wikihow_text_category":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, category = data["text"], data["category"]
                    yield id_, {
                        "text": text,
                        "category": category
                    }

        elif self.config.name == "wikihow_category_hierarchy":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    high_category, low_category = data["high_category"], data["low_category"]
                    yield id_, {
                        "high_category": high_category,
                        "low_category": low_category
                    }

        elif self.config.name == "wikihow_goal_step":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    goal, steps = data["goal"], data["steps"]
                    yield id_, {
                        "goal": goal,
                        "steps": steps
                    }

        elif self.config.name == "wikihow_summary":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, summary = data["text"], data["summary"]
                    yield id_, {
                        "text": text,
                        "summary": summary
                    }

        elif self.config.name == "wikihow_procedure":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    goal, first_step, second_step = data["goal"], data["first_step"], data["second_step"]
                    yield id_, {
                        "goal": goal,
                        "first_step": first_step,
                        "second_step": second_step
                    }

        elif self.config.name == "wikihow_question":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    question, description, answer, related_questions = data["question"], data["description"], data[
                        "answer"], data["related_questions"]
                    yield id_, {
                        "question": question,
                        "description": description,
                        "answer": answer,
                        "related_questions": related_questions
                    }

        elif self.config.name == "wikipedia_entities":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    try:
                        data = json.loads(row.strip())
                    except json.decoder.JSONDecodeError:
                        data = eval(row.strip())
                    text, entities = data["text"], data["entities"]
                    yield id_, {
                        "text": text,
                        "entities": entities
                    }

        elif self.config.name == "wikipedia_sections":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    texts, titles = data["texts"], data["titles"]
                    yield id_, {
                        "texts": texts,
                        "titles": titles
                    }

        elif self.config.name == "wordnet_pos":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    word, sentence, pos = data["word"], data["sentence"], data["pos"]
                    yield id_, {
                        "word": word,
                        "sentence": sentence,
                        "pos": pos
                    }

        elif self.config.name == "wordnet_meaning":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    word, sentence, meaning, possible_meanings = data["word"], data["sentence"], data["meaning"], data[
                        "possible_meanings"]
                    yield id_, {
                        "word": word,
                        "sentence": sentence,
                        "meaning": meaning,
                        "possible_meanings": possible_meanings
                    }

        elif self.config.name == "wordnet_synonym":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    word, sentence, synonyms = data["word"], data["sentence"], data["synonyms"]
                    yield id_, {
                        "word": word,
                        "sentence": sentence,
                        "synonyms": synonyms
                    }

        elif self.config.name == "wordnet_antonym":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    word, sentence, antonyms = data["word"], data["sentence"], data["antonyms"]
                    yield id_, {
                        "word": word,
                        "sentence": sentence,
                        "antonyms": antonyms
                    }

        elif self.config.name == "qa_control":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    premise, hypothesis, label = data["premise"], data["hypothesis"], data["label"]
                    yield id_, {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label
                    }

        elif self.config.name == "qa_dream":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    context, question, options, answer = data["text"], data["question"], data["options"], data["answer"]
                    yield id_, {
                        "context": context,
                        "question": question,
                        "options": options,
                        "answer": answer
                    }

        elif self.config.name in ["qa_logiqa", "qa_reclor", "qa_race", "qa_race_c"]:
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    context, question, options, answer = data["context"], data["question"], data["options"], data[
                        "answer"]
                    yield id_, {
                        "context": context,
                        "question": question,
                        "options": options,
                        "answer": answer
                    }

        elif self.config.name == "qa_triviaqa":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    context, question, answer = data["context"], data["question"], data["answer"]
                    yield id_, {
                        "context": context,
                        "question": question,
                        "answer": answer
                    }

        elif self.config.name == "arxiv_category":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, category = data["text"], data["category"]
                    yield id_, {
                        "text": text,
                        "category": category
                    }

        elif self.config.name == "arxiv_summary":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, summary = data["text"], data["summary"]
                    yield id_, {
                        "text": text,
                        "summary": summary
                    }

        elif self.config.name == "paperswithcode_entity":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, entities, datasets, methods, tasks, metrics = data["text"], data["entities"], data[
                        "datasets"], data["methods"], data["tasks"], data["metrics"]
                    yield id_, {
                        "text": text,
                        "entities": entities,
                        "datasets": datasets,
                        "methods": methods,
                        "tasks": tasks,
                        "metrics": metrics
                    }

        elif self.config.name == "paperswithcode_summary":
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row.strip())
                    text, summary = data["text"], data["summary"]
                    yield id_, {
                        "text": text,
                        "summary": summary
                    }
