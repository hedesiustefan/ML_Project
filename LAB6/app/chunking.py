import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.logger import logger

log = logger()


def chunk_documents(
    input_path="data/processed_data/docs.jsonl",
    output_path="data/chunks/chunks.jsonl",
    chunk_size=800,
    chunk_overlap=150,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for doc_id, line in enumerate(fin):
            doc = json.loads(line)
            text = doc.get("text", "").strip()
            source = doc.get("source", "unknown")

            if not text:
                log.warning(f"[chunking] empty document skipped: {source}")
                continue

            chunks = splitter.split_text(text)

            for chunk_id, chunk_text in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc_id}-{chunk_id}",
                    "source": source,
                    "text": chunk_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            log.info(
                f"[chunking] {source} → {len(chunks)} chunks"
            )

    log.info(
        f"[chunking] completed: {total_chunks} total chunks written to {output_path}"
    )


if __name__ == "__main__":
    chunk_documents()
