"""Delegation uses the same unlimited spellings as the parent agent."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task


@pytest.mark.parametrize("configured,expected", [(0, sys.maxsize), ("unlimited", sys.maxsize), (37, 37)])
def test_configured_budget_reaches_child_constructor(configured, expected):
    parent = _make_mock_parent(depth=0)
    with patch("tools.delegate_tool._load_config", return_value={"max_iterations": configured}), \
            patch("run_agent.AIAgent") as factory:
        child = MagicMock()
        child.run_conversation.return_value = {"final_response": "done", "completed": True, "api_calls": 1}
        factory.return_value = child
        delegate_task(goal="Inspect the saved state", parent_agent=parent)
        assert factory.call_args.kwargs["max_iterations"] == expected
