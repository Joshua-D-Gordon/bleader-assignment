"""Lazy singletons for the OpenAI and Pinecone clients.

Constructed on first use so importing modules (e.g. for parser unit tests) never
requires API keys — keys are only demanded when a client is actually called.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def openai_client():
    from openai import OpenAI
    config.require_keys("OPENAI_API_KEY")
    return OpenAI(api_key=config.OPENAI_API_KEY)


@lru_cache(maxsize=1)
def pinecone_client():
    from pinecone import Pinecone
    config.require_keys("PINECONE_API_KEY")
    return Pinecone(api_key=config.PINECONE_API_KEY)
