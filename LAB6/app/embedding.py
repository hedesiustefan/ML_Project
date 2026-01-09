import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from app.logger import logger

log = logger()


def create_embeddings(
    chunks_path="data/chunks/chunks.jsonl",
    index_path="data/processed_data/faiss.index",
    metadata_path="data/chunks/chunks_meta.json",
    model_name="models/minilm-finetuned",
):
    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    texts = []
    metadata = []


    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metadata.append({
                "chunk_id": rec["chunk_id"],
                "source": rec["source"],
            })

    log.info(f"[embedding] loaded {len(texts)} chunks")

    
    model = SentenceTransformer(model_name, device="cuda")

    
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    log.info(f"[embedding] embeddings shape: {embeddings.shape}")

    # build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # save index + metadata
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    log.info(f"[embedding] FAISS index saved to {index_path}")
    log.info(f"[embedding] metadata saved to {metadata_path}")


if __name__ == "__main__":
    create_embeddings()
