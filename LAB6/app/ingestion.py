import json
from pathlib import Path
from PyPDF2 import PdfReader
from app.logger import logger

log = logger()

def clean_text(text: str) -> str:
    # to remove invalid characters
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        pages.append(txt)
        log.info(f"[ingest] read page {i+1}/{len(reader.pages)} from {path.name}")
    return "\n".join(pages)


def ingest_folder(input_dir="data/raw_data", output_path="data/processed_data/docs.jsonl"):
    input_dir = Path(input_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    docs = []
    for p in input_dir.rglob("*"):
        text = read_pdf(p)

        docs.append({"source": str(p), "text": text})
        log.info(f"[ingest] loaded {p}")

    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    log.info(f"[ingest] wrote {len(docs)} docs to {out}")

if __name__ == "__main__":
    ingest_folder()
