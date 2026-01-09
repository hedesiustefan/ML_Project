import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from app.logger import logger

log = logger()

def load_chunks(path, limit=3000):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            texts.append(json.loads(line)["text"])
    return texts


def finetune_minilm(
    chunks_path="data/chunks/chunks.jsonl",
    output_dir="models/minilm-finetuned",
    batch_size=32,
    epochs=1,
):
    texts = load_chunks(chunks_path)
    log.info(f"[finetune] loaded {len(texts)} chunks")

    # create positive pairs (chunk, chunk)
    train_examples = [
        InputExample(texts=[text, text])
        for text in texts
    ]

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )

    loss = MultipleNegativesRankingLoss(model)

    log.info("[finetune] starting training...")
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        show_progress_bar=True,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    log.info(f"[finetune] model saved to {output_dir}")


if __name__ == "__main__":
    finetune_minilm()
