import os
from typing import List, Tuple

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


class DocumentLoader:
    @staticmethod
    def load_pdf(filepath: str) -> List[Tuple[str, dict]]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files. Install it using: pip install pypdf"
            )

        reader = PdfReader(filepath)
        pages_content: List[Tuple[str, dict]] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                metadata = {
                    "source": filepath,
                    "filename": os.path.basename(filepath),
                    "page_number": page_idx,
                    "total_pages": len(reader.pages),
                }
                pages_content.append((text, metadata))

        return pages_content

    @staticmethod
    def load_text_file(filepath: str) -> List[Tuple[str, dict]]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = {
            "source": filepath,
            "filename": os.path.basename(filepath),
        }
        return [(content, metadata)]

    @classmethod
    def load_file(cls, filepath: str) -> List[Tuple[str, dict]]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return cls.load_pdf(filepath)
        elif ext in (".txt", ".md"):
            return cls.load_text_file(filepath)
        else:
            raise ValueError(f"Unsupported file extension '{ext}' for file: {filepath}")

    @classmethod
    def find_supported_files(cls, path: str, recursive: bool = False) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            return [path] if ext in SUPPORTED_EXTENSIONS else []

        matched_files: List[str] = []
        if recursive:
            for root, _, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                        matched_files.append(os.path.join(root, file))
        else:
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in SUPPORTED_EXTENSIONS:
                    matched_files.append(full_path)

        return sorted(matched_files)
