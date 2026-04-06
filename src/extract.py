"""Extract clean text from book files (txt, epub, pdf)."""
import os
from pathlib import Path

SUPPORTED_FORMATS = {".txt", ".epub", ".pdf"}


def extract_text(file_path: str) -> str:
    """Extract text from a single file. Returns empty string on failure."""
    path = Path(file_path)

    if not path.exists():
        return ""

    ext = path.suffix.lower()

    if ext == ".txt":
        return _extract_txt(path)
    elif ext == ".epub":
        return _extract_epub(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    else:
        return ""


def extract_from_directory(dir_path: str) -> list[dict]:
    """Extract text from all supported files in a directory.

    Returns a list of dicts: [{"source": "filename.txt", "text": "..."}]
    """
    results = []
    dir_path = Path(dir_path)

    if not dir_path.exists():
        return results

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_FORMATS:
            text = extract_text(str(file_path))
            if text.strip():
                results.append({
                    "source": file_path.name,
                    "text": text,
                })

    return results


def _extract_txt(path: Path) -> str:
    """Read a plain text file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _extract_epub(path: Path) -> str:
    """Extract text from an EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        print("WARNING: ebooklib/beautifulsoup4 not installed. Skipping EPUB.")
        return ""

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        text = text.strip()
        if text:
            chapters.append(text)

    return "\n\n".join(chapters)


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("WARNING: PyPDF2 not installed. Skipping PDF.")
        return ""

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)
