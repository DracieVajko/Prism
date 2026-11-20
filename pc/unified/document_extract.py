"""Extract text from documents (txt, md, docx, pdf) and images (OCR).

Used by /v1/documents/extract. PDF text extraction uses pdfminer when the PDF
has a text layer; scanned (image-only) PDFs fall back to rendering each page
with pypdfium2 and running the installed OCR engine on it.
"""
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _extract_docx(path: Path) -> str:
    """Extract paragraph text from a .docx without python-docx."""
    try:
        with zipfile.ZipFile(path) as z:
            xml_bytes = z.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError):
        return ""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    parts = []
    for p in root.iter(f"{ns}p"):
        texts = [t.text or "" for t in p.iter(f"{ns}t")]
        line = "".join(texts).strip()
        if line:
            parts.append(line)
    return "\n".join(parts)


def _extract_pdf_text(path: Path, password: str = "") -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""
    try:
        return extract_text(str(path), password=password) or ""
    except Exception:
        return ""


def _ocr_image_path(path: Path, language: str = "auto") -> str:
    try:
        from unified.ocr_engine import get_ocr_engine
    except Exception:
        return ""
    try:
        engine = get_ocr_engine()
        if not engine.is_installed() or not engine.init():
            return ""
        return engine.transcribe(str(path), language=language) or ""
    except Exception:
        return ""


def _extract_pdf(path: Path, password: str = "", language: str = "auto"):
    """Return (text, method, page_count)."""
    text = _extract_pdf_text(path, password=password)
    method = "pdfminer"
    page_count = 0
    # Render pages for OCR fallback if text is (almost) empty.
    pdf = None
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path), password=password) if password else pdfium.PdfDocument(str(path))
        page_count = len(pdf)
    except Exception:
        pdf = None
        page_count = 0

    if len(text.strip()) < 50 and pdf is not None:
        # scanned PDF -> OCR each page
        ocr_parts = []
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                bmp = page.render(scale=2.0)
                img = bmp.to_pil()
                tmp = Path(tempfile.mktemp(suffix=".png"))
                img.save(tmp)
                try:
                    page_text = _ocr_image_path(tmp, language=language)
                finally:
                    tmp.unlink(missing_ok=True)
                if page_text.strip():
                    ocr_parts.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
            if ocr_parts:
                text = "\n\n".join(ocr_parts)
                method = "ocr"
        except Exception:
            pass
    try:
        if pdf is not None:
            pdf.close()
    except Exception:
        pass
    return text, method, page_count


def extract(path: Path, password: str = "", language: str = "auto") -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown", ".csv", ".log", ".rtf"):
        return {"text": _read_text_file(path), "method": "text", "pages": 0}
    if suffix == ".docx":
        return {"text": _extract_docx(path), "method": "docx-xml", "pages": 0}
    if suffix == ".pdf":
        text, method, pages = _extract_pdf(path, password=password, language=language)
        return {"text": text, "method": method, "pages": pages}
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        return {"text": _ocr_image_path(path, language=language), "method": "ocr", "pages": 1}
    # Fallback: try reading as text.
    try:
        return {"text": _read_text_file(path), "method": "text", "pages": 0}
    except Exception:
        return {"text": "", "method": "none", "pages": 0}


def extract_bytes(data: bytes, filename: str, password: str = "",
                   language: str = "auto") -> dict:
    suffix = Path(filename).suffix.lower() or ".bin"
    tmp = Path(tempfile.mktemp(suffix=suffix))
    try:
        tmp.write_bytes(data)
        return extract(tmp, password=password, language=language)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            import time
            time.sleep(0.5)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
