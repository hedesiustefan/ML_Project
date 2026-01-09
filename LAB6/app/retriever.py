import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.logger import logger

log = logger()


class Retriever:
    def __init__(
        self,
        index_path="data/processed_data/faiss.index",
        chunks_path="data/chunks/chunks.jsonl",
        model_name="models/minilm-finetuned",
        device="cuda",
    ):
        # FAISS index file (stores the embeddings)
        self.index = faiss.read_index(index_path)

        # load chunks +  lookup by FAISS index position
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        
        assert len(self.chunks) == self.index.ntotal, (
            "Mismatch between FAISS vectors and chunks"
        )

        # embedding model
        self.embedder = SentenceTransformer(model_name, device=device)

        log.info(f"[retriever] loaded {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 5):
        # embed query
        query_vec = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search
        scores, indices = self.index.search(query_vec[None, :], top_k)

        # maps FAISS indices to the chunk text
        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
            })

        return results
