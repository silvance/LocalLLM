import re
from dataclasses import dataclass

from app.schemas.chat import ChatRequest
from app.schemas.model import ModelKey


@dataclass
class RoutingDecision:
    selected_model: ModelKey
    reason: str
    complexity_score: int


CODE_FENCE_PATTERN = re.compile(r"```")
FILE_EXTENSION_PATTERN = re.compile(
    r"\.(py|js|ts|tsx|jsx|rs|cpp|cc|c|h|hpp|go|java|cs|rb|php|swift|kt|scala|sh|ps1|sql|yaml|yml|toml|json|html|css)\b",
    re.IGNORECASE,
)
CODE_KEYWORDS = {
    "code", "function", "method", "class", "implement", "debug", "refactor",
    "syntax", "compile", "stacktrace", "stack trace", "traceback", "exception",
    "regex", "algorithm", "endpoint", "schema", "query",
    "framework", "package", "dependency", "snippet",
    "bug", "crash", "unit test", "type hint",
    "python", "javascript", "typescript", "rust", "golang", "java", "c++",
    "ruby", "powershell", "bash", "node", "react",
}
COMPLEX_KEYWORDS = {
    "analyze", "compare", "evaluate", "reason", "tradeoffs", "architecture",
    "design", "strategy", "multi-step", "step-by-step", "plan", "why",
}
SIMPLE_KEYWORDS = {
    "hello", "hi", "hey", "summarize", "rewrite", "fix grammar",
    "one sentence", "short answer", "what is", "who is",
}


def _count_word_matches(text: str, keywords: set[str]) -> int:
    matches = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            matches += 1
    return matches


class ModelRouter:
    def route(self, request: ChatRequest) -> RoutingDecision:
        if not request.messages:
            return RoutingDecision(
                selected_model="granite",
                reason="No messages; defaulting to fast model.",
                complexity_score=0,
            )

        latest_message = request.messages[-1].content
        latest_lower = latest_message.lower()
        word_count = len(latest_lower.split())

        reasons: list[str] = []
        code_score = 0
        complexity_score = 0

        if CODE_FENCE_PATTERN.search(latest_message):
            code_score += 3
            reasons.append("Code block.")
        if FILE_EXTENSION_PATTERN.search(latest_message):
            code_score += 2
            reasons.append("File extension.")

        code_kw_matches = _count_word_matches(latest_lower, CODE_KEYWORDS)
        if code_kw_matches >= 2:
            code_score += 2
            reasons.append(f"{code_kw_matches} code keywords.")
        elif code_kw_matches == 1:
            code_score += 1
            reasons.append("1 code keyword.")

        if word_count > 40:
            complexity_score += 2
            reasons.append("Long prompt.")
        elif word_count > 20:
            complexity_score += 1
            reasons.append("Moderately long prompt.")

        complex_kw_matches = _count_word_matches(latest_lower, COMPLEX_KEYWORDS)
        if complex_kw_matches >= 1:
            complexity_score += min(complex_kw_matches, 2)
            reasons.append(f"{complex_kw_matches} complexity keyword(s).")

        simple_kw_matches = _count_word_matches(latest_lower, SIMPLE_KEYWORDS)
        if simple_kw_matches >= 1:
            complexity_score -= 1
            reasons.append("Simple-task keyword.")

        if "?" in latest_lower and word_count > 25:
            complexity_score += 1
            reasons.append("Long-form question.")
        if latest_lower.count("\n") >= 2:
            complexity_score += 1
            reasons.append("Multi-part formatting.")

        total_score = code_score + complexity_score
        reason_text = " ".join(reasons) if reasons else "Default routing."

        if code_score >= 1:
            return RoutingDecision(
                selected_model="qwen",
                reason=reason_text,
                complexity_score=total_score,
            )

        if complexity_score >= 2:
            return RoutingDecision(
                selected_model="gemma",
                reason=reason_text,
                complexity_score=total_score,
            )

        return RoutingDecision(
            selected_model="granite",
            reason=reason_text,
            complexity_score=total_score,
        )
