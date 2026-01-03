from app.retriever import Retriever

retriever = Retriever()
results = retriever.retrieve('What is a cost function')

for r in results:
        print("\n---")
        print("Score:", r["score"])
        print(r["text"][:300])