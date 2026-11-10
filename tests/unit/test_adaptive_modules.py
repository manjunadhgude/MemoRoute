from datetime import datetime

import pytest

from memoroute.retrieval.context_packet import estimate_tokens
from memoroute.retrieval.query_intent import QueryPurpose, infer_query_intent
from memoroute.retrieval.temporal_selector import TemporalQueryMode


def test_plain_query_defaults_to_unspecified():
    from memoroute.retrieval.query_intent import QueryPurpose
    result = infer_query_intent("Find the deployment configuration")
    assert result.mode == TemporalQueryMode.UNSPECIFIED
    assert result.purpose == QueryPurpose.LOOKUP

def test_comparison_is_detected():
    result = infer_query_intent("Compare production versus staging")
    assert result.purpose == QueryPurpose.COMPARISON

def test_current_mode_requires_aware_time():
    with pytest.raises(TypeError):
        infer_query_intent("current deployment", mode=TemporalQueryMode.CURRENT)
    with pytest.raises(ValueError):
        infer_query_intent("current deployment", mode=TemporalQueryMode.CURRENT,
                           as_of=datetime(2026, 11, 4))

def test_token_estimate_is_deterministic():
    assert estimate_tokens("one two three") == 4
