from enum import Enum
from typing import TypedDict


class Language(str, Enum):
    KOR = "kor"
    ENG = "eng"


class BaseState(TypedDict):
    language: Language
