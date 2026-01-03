from app.retriever import Retriever
from app.generator import Generator

retriever = Retriever()
generator = Generator()


def build_rag_prompt(question: str, retrieved_chunks: list[str]) -> str:
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



def answer_question(question: str):
    retrieved = retriever.retrieve(question, top_k=5)
    prompt = build_rag_prompt(question, retrieved)
    answer = generator.generate(prompt)
    return answer, retrieved

def main():
    answer, chunks = answer_question("What is a cost function?")

if __name__ == "__main__":
    main()
