from __future__ import annotations

import abc
from functools import lru_cache
from typing import List, Optional

import jieba

tokenizer_registry = {}


def register_tokenizer(name: str):
    """
    register for tokenizer
    """

    def register_tokenizer_cls(cls):
        tokenizer_registry[name] = cls
        return cls

    return register_tokenizer_cls


def get_default_tokenizer(
    task_type: Optional[str] = None, language: Optional[str] = None
):
    if task_type is None or language is None:
        return SingleSpaceTokenizer()
    if language == "zh":
        return JiebaTokenizer()
    else:
        return SingleSpaceTokenizer()


def get_tokenizer(
    tokenizer_name: Optional[str] = None, task_type: str = None, language: str = None
):

    if tokenizer_name is None:
        return get_default_tokenizer(task_type, language)
    else:
        if tokenizer_name not in tokenizer_registry.keys():
            raise ValueError(f"{tokenizer_name} is not supported")
        else:
            return tokenizer_registry[tokenizer_name]()


class Tokenizer:
    @abc.abstractmethod
    def __call__(self, text: str) -> list[str]:
        """
        virtual base class of tokenizer
        """
        ...


@register_tokenizer("SingleSpaceTokenizer")
class SingleSpaceTokenizer(Tokenizer):
    """
    Tokenize a string based on the space
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> List[str]:
        return text.split(" ")


@register_tokenizer("JiebaTokenizer")
class JiebaTokenizer(Tokenizer):
    """
    Tokenizer a string using Jieba segmentor
    """

    @lru_cache(maxsize=20)
    def __call__(self, text: str) -> List[str]:
        # TODO(Pengfei): this should be optimized
        return [w for w in jieba.cut(text, cut_all=False)]
