"""Anchor prefilter contracts, including changed-pattern and unusual-content controls."""
from __future__ import annotations
import re
import pytest
from agent.compression_anchor_candidates import anchor_candidate_text
from agent.context_compressor import _build_anchor_index,_ANCHOR_PATTERNS


@pytest.mark.parametrize('label,pattern,_cap',_ANCHOR_PATTERNS)
def test_large_unicode_transcript_retains_exact_matches_and_order(label,pattern,_cap):
    # Full-size, tool-heavy shape with CRLF, Unicode paths, ties and opaque data.
    noise='opaque payload 12345678 λ雪\r\n'*10000
    interesting='src/雪.py\r\nfix/caller-path #1234 deadbeef01234 @worker\nhttps://example.com/receipt/731904\nValueError: receipt absent\n'
    text=noise+interesting+noise+interesting
    before=[m.group(0) for m in pattern.finditer(text)]
    after=[m.group(0) for m in pattern.finditer(anchor_candidate_text(text,pattern.pattern))]
    assert after==before


def test_unknown_expression_keeps_its_newline_spanning_matches():
    pattern=re.compile(r'BEGIN\nvalue\nEND')
    text='unrelated\nBEGIN\nvalue\nEND\nunrelated'
    assert pattern.findall(anchor_candidate_text(text,pattern.pattern))==['BEGIN\nvalue\nEND']


def test_dense_single_line_and_non_text_rows_keep_public_anchor_output():
    rows=[{'content':None},{'content':{'type':'image'}},{'content':'src/雪.py src/雪.py fix/caller-path #1234 deadbeef01234'}]
    result=_build_anchor_index(rows)
    assert 'files: src/雪.py(x2)' in result
    assert 'branches: fix/caller-path' in result
    assert 'commits: deadbeef01234' in result
    assert 'PRs/issues: #1234' in result
