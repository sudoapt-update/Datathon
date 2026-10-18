"""Extract text from the official rule PDFs."""

from pathlib import Path

from pypdf import PdfReader

FILES = [
    Path(r"C:\Users\User\Downloads\Об_утверждении_Правил_осуществления_кассовых_опера.pdf"),
    Path(r"C:\Users\User\Downloads\Об_утверждении_Правил_осуществления_обменных_опера.pdf"),
    Path(r"C:\Users\User\Downloads\О_противодействии_легализации_отмыванию_доходов_.pdf"),
]
OUT = Path(r"C:\Users\User\Documents\Datathon\data\rules_text")
OUT.mkdir(parents=True, exist_ok=True)

for path in FILES:
    reader = PdfReader(str(path))
    parts = []
    for index, page in enumerate(reader.pages, start=1):
        parts.append(f"\n\n===== PAGE {index} =====\n")
        parts.append(page.extract_text() or "")
    dest = OUT / (path.stem[:40] + ".txt")
    dest.write_text("".join(parts), encoding="utf-8")
    print(dest.name, len(reader.pages), dest.stat().st_size)
