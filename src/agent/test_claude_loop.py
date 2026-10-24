"""
Tests the Claude tool-use loop WITHOUT an API key, using a fake client that replays
what the real API returns. This checks the plumbing (tool dispatch, tool_result format,
loop exit) - not the quality of Claude's answers.

    python -m src.agent.test_claude_loop
"""
from types import SimpleNamespace as NS

from src.agent.agent import run_claude


class FakeClient:
    def __init__(self):
        self.calls, self.messages = 0, NS(create=self.create)
        self.seen_tool_results = []

    def create(self, model, max_tokens, system, tools, messages):
        self.calls += 1
        assert any(t["name"] == "sales_trend" for t in tools), "tool specs must be passed"
        if self.calls == 1:   # Claude asks for two tools at once
            return NS(stop_reason="tool_use", content=[
                NS(type="text", text="Let me check the trend and churn."),
                NS(type="tool_use", id="t1", name="sales_trend", input={"region": "West"}),
                NS(type="tool_use", id="t2", name="churn_risk", input={"region": "West", "top_n": 2})])
        last = messages[-1]
        assert last["role"] == "user" and all(r["type"] == "tool_result" for r in last["content"])
        self.seen_tool_results = last["content"]
        return NS(stop_reason="end_turn", content=[NS(type="text", text="## Findings\n- West fell.\n## Recommended actions\n1. Act.")])


def main():
    fake = FakeClient()
    out = run_claude("Why are West sales falling?", client=fake, save_report=False)
    assert fake.calls == 2
    assert out["tools_used"] == ["sales_trend", "churn_risk"]
    ids = [r["tool_use_id"] for r in fake.seen_tool_results]
    assert ids == ["t1", "t2"], ids
    assert "Rs" in fake.seen_tool_results[0]["content"]          # real tool output was sent back
    assert "## Findings" in out["report"]
    print("Claude tool-use loop: OK (2 API calls, 2 tools run, results returned with matching ids)")


if __name__ == "__main__":
    main()
