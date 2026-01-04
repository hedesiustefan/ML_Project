import re

BANNED_PATTERNS = [
    r"\b(kill|murder|bomb|terror)\b",
    r"\b(porn|sexual|rape)\b",
    r"\b(hate|racist)\b",
]

def violates_policy(text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in BANNED_PATTERNS)


def input_guardrail(question: str) -> tuple[bool, str | None]:
    if violates_policy(question):
        return False, "The question violates content policy."

    if not any(k in question.lower() for k in ["loss", "model", "learning", "neural", "function"]):
        return False, "The question is off-topic for the provided documents."

    return True, None

def output_guardrail(answer: str) -> tuple[bool, str | None]:
    if violates_policy(answer):
        return False, "Generated answer violates content policy."
    return True, None
