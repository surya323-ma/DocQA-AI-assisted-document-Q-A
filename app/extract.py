"""Pull plain text out of uploaded files."""
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError

ALLOWED = {".pdf", ".txt", ".md"}


class ExtractError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    ext = name[name.rfind("."):] if "." in name else ""
    if ext not in ALLOWED:
        raise ExtractError("Upload a PDF, TXT or MD file.")
    if ext == ".pdf":
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ExtractError("That PDF is password-protected.")
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except (PyPdfError, ValueError, OSError) as exc:
            raise ExtractError("That PDF could not be read.") from exc
    else:
        text = data.decode("utf-8", errors="replace")
    if not text.strip():
        raise ExtractError("No readable text found. Scanned PDFs need OCR first.")
    return text
