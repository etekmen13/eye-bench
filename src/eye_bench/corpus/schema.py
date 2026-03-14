from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    message_id: str
    length: int
    symbols: list[int]


class Corpus(BaseModel):
    alphabet_size: int = Field(gt=0)
    messages: list[Message]
    messages_unigram: list[Message]
