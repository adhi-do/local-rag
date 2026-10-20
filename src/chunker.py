from dataclasses import dataclass, field
from typing import List, Dict, Any
import re


@dataclass
class Chunk:
    text: str
    chunk_id: int
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split text on paragraph boundaries or sentence terminal punctuation
        raw_units = re.split(r"(?<=[.!?])\s+|\n\n+", text.strip())
        return [unit.strip() for unit in raw_units if unit.strip()]

    def split_text(self, text: str, source: str = "") -> List[Chunk]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_id = 0

        for sentence in sentences:
            sent_len = len(sentence)

            # If adding the sentence exceeds chunk_size, finalize the current chunk
            if current_sentences and (current_len + sent_len + 1 > self.chunk_size):
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={"character_count": len(chunk_text)},
                    )
                )
                chunk_id += 1

                # Carry over trailing sentences to create overlap
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sent_len + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={"character_count": len(chunk_text)},
                )
            )

        return chunks
