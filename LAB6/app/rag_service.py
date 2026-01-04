from app.retriever import Retriever
from app.generator import Generator
from app.guardrails import input_guardrail, output_guardrail
from app.judge import LLMJudge

retriever = Retriever()
generator = Generator()
judge = LLMJudge()


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return (
        "You are a helpful assistant.\n"
        "Answer the question using ONLY the information in the context below.\n"
        "If the answer is not contained in the context, say:\n"
        "'I don't know based on the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def answer_question(question: str) -> dict:
    # input guardrail
    ok, msg = input_guardrail(question)
    if not ok:
        return {"error": msg}

    # rag part
    retrieved = retriever.retrieve(question, top_k=5)
    prompt = build_rag_prompt(question, retrieved)
    answer = generator.generate(prompt)

    # output guardrail
    ok, msg = output_guardrail(answer)
    if not ok:
        return {"error": msg}

    # judge
    context = "\n\n".join(c["text"] for c in retrieved)
    verdict = judge.evaluate(question, context, answer)

    return {
        "answer": answer,
        "verdict": verdict,
        "sources": [
            {"source": c["source"], "score": c.get("score")}
            for c in retrieved
        ]
    }
