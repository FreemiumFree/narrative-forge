"""Extract clean text from book files (txt, epub, pdf, mobi)."""
import os
from pathlib import Path

SUPPORTED_FORMATS = {".txt", ".epub", ".pdf", ".mobi"}


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
    elif ext == ".mobi":
        return _extract_mobi(path)
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
            try:
                text = extract_text(str(file_path))
                if text.strip():
                    results.append({
                        "source": file_path.name,
                        "text": text,
                    })
            except Exception as e:
                print(f"  WARNING: Failed to extract {file_path.name}: {e}")

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


def _extract_mobi(path: Path) -> str:
    """Extract text from a MOBI file by converting to EPUB first."""
    try:
        import mobi
    except ImportError:
        print("WARNING: mobi not installed. Skipping MOBI.")
        return ""

    # mobi.extract returns (temp_dir, epub_path)
    temp_dir, epub_path = mobi.extract(str(path))
    epub_file = Path(epub_path)

    # Try epub extraction first
    try:
        result = _extract_epub(epub_file)
        if result.strip():
            return result
    except Exception:
        pass

    # Fallback: parse any HTML files in the temp directory
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    temp = Path(temp_dir)
    html_files = sorted(temp.rglob("*.html")) + sorted(temp.rglob("*.htm"))
    chapters = []
    for html_file in html_files:
        try:
            html = html_file.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            text = text.strip()
            if text:
                chapters.append(text)
        except Exception:
            continue

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
