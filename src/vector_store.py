from dataclasses import dataclass, field
import json
import math
from typing import List, Dict, Any, Tuple


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions must match: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


@dataclass
class VectorRecord:
    record_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, record_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
        record = VectorRecord(
            record_id=record_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.records.append(record)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[VectorRecord, float]]:
        if not self.records:
            return []

        scored_records = []
        for record in self.records:
            score = cosine_similarity(query_embedding, record.embedding)
            scored_records.append((record, score))

        scored_records.sort(key=lambda item: item[1], reverse=True)
        return scored_records[:top_k]

    def count(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()

    def save(self, filepath: str) -> None:
        data = [
            {
                "record_id": r.record_id,
                "text": r.text,
                "embedding": r.embedding,
                "metadata": r.metadata,
            }
            for r in self.records
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.records = [
            VectorRecord(
                record_id=item["record_id"],
                text=item["text"],
                embedding=item["embedding"],
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
