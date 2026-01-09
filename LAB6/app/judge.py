import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY is not None


class LLMJudge:
    def __init__(self, model_name="xiaomi/mimo-v2-flash:free"):
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def evaluate(self, question: str, context: str, answer: str) -> dict:
        prompt = (
            "You are an evaluator.\n"
            "Evaluate the ANSWER based strictly on the CONTEXT.\n"
            "Return a JSON object with the following fields:\n"
            "- groundedness (0-10)\n"
            "- completeness (0-10)\n"
            "- hallucination_risk (low/medium/high)\n"
            "- brief_feedback\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "JSON:\n"
        )

        print('Sys prompt for judge loaded')
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompt}
                ],
                "temperature": 0.0
            },
            timeout=30,
        )
        print('Request for judge made')

        response.raise_for_status()
        data = response.json()

        # extract model text
        content = data["choices"][0]["message"]["content"]

        return content
