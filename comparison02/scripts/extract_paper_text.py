from __future__ import annotations

from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover
    extract_text = None


def main() -> None:
    pdf_path = Path(__file__).resolve().parents[1] / "基于全卷积残差收缩网络的地震波阻抗反演_王康.pdf"
    out_path = Path(__file__).resolve().parents[1] / "paper_extracted.txt"

    text = ""
    if fitz is not None:
        doc = fitz.open(str(pdf_path))
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        text = "\n".join(parts)
    elif extract_text is not None:
        # pdfminer can be slow on some PDFs; keep as fallback only.
        text = extract_text(str(pdf_path))
    else:
        raise RuntimeError("Neither PyMuPDF nor pdfminer.six is available for PDF text extraction.")
    out_path.write_text(text, encoding="utf-8")

    print(f"[OK] extracted chars={len(text)}")
    print(f"[OK] wrote: {out_path}")
    print("--- head ---")
    print(text[:2000])


if __name__ == "__main__":
    main()
