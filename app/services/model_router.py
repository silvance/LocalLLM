from dataclasses import dataclass

from app.schemas.chat import ChatRequest
from app.schemas.model import ModelKey


@dataclass
class RoutingDecision:
    selected_model: ModelKey
    reason: str
    complexity_score: int


class ModelRouter:
    COMPLEX_KEYWORDS = {
        "analyze",
        "compare",
        "evaluate",
        "reason",
        "explain deeply",
        "tradeoffs",
        "architecture",
        "design",
        "strategy",
        "multi-step",
        "step-by-step",
        "debug",
        "refactor",
        "optimize",
        "plan",
        "implement",
        "why",
    }

    SIMPLE_KEYWORDS = {
        "hello",
        "hi",
        "summarize",
        "rewrite",
        "fix grammar",
        "one sentence",
        "short answer",
        "what is",
        "who is",
    }

    def route(self, request: ChatRequest) -> RoutingDecision:
        if not request.messages:
            return RoutingDecision(
                selected_model="granite",
                reason="No messages found; defaulting to fast model.",
                complexity_score=0,
            )

        latest_message = request.messages[-1].content.strip().lower()
        score = 0
        reasons: list[str] = []

        word_count = len(latest_message.split())

        if word_count > 40:
            score += 2
            reasons.append("Prompt is long.")
        elif word_count > 20:
            score += 1
            reasons.append("Prompt is moderately long.")

        if any(keyword in latest_message for keyword in self.COMPLEX_KEYWORDS):
            score += 2
            reasons.append("Complexity keywords detected.")

        if any(keyword in latest_message for keyword in self.SIMPLE_KEYWORDS):
            score -= 1
            reasons.append("Simple-task keywords detected.")

        if "?" in latest_message and word_count > 25:
            score += 1
            reasons.append("Long-form question detected.")

        if latest_message.count("\n") >= 2:
            score += 1
            reasons.append("Multi-part formatting detected.")

        if score >= 2:
            return RoutingDecision(
                selected_model="gemma",
                reason=" ".join(reasons) if reasons else "Complex request detected.",
                complexity_score=score,
            )

        return RoutingDecision(
            selected_model="granite",
            reason=" ".join(reasons) if reasons else "Simple request detected.",
            complexity_score=score,
        )