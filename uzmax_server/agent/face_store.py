import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class FaceVectorStore:
    def __init__(
        self,
        db_path: str = "data/faces/qdrant",
        collection_name: str = "people_faces",
        vector_size: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._repair_local_meta_compatibility()
        self.client = QdrantClient(path=str(self.db_path))

    def _repair_local_meta_compatibility(self) -> None:
        meta_path = self.db_path / "meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return

        changed = False
        for collection in (meta.get("collections") or {}).values():
            for key in ("strict_mode_config", "metadata"):
                if key in collection:
                    collection.pop(key, None)
                    changed = True

        if changed:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
        except Exception:
            return False
        return any(collection.name == self.collection_name for collection in collections)

    @staticmethod
    def _normalize(embedding: list[float]) -> list[float]:
        arr = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            info = self.client.get_collection(self.collection_name)
            config_size = info.config.params.vectors.size
            if config_size != vector_size:
                self.client.delete_collection(self.collection_name)
            else:
                return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def identify(self, embedding: list[float], threshold: float = 0.62, margin: float = 0.04) -> dict | None:
        if not embedding:
            return None

        normalized = self._normalize(embedding)
        self._ensure_collection(len(normalized))

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        comparisons = []
        people = {}
        best_record = None
        best_score = -1.0

        for record in records:
            vector = record.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()))
            if vector is None:
                continue
            if len(vector) != len(normalized):
                continue

            score = self._cosine_similarity(normalized, vector)
            payload = record.payload or {}
            person_id = payload.get("person_id") or str(record.id)
            comparison = {
                "person_id": person_id,
                "first_name": payload.get("first_name", ""),
                "last_name": payload.get("last_name", ""),
                "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
                "score": round(score, 4),
                "metadata": payload.get("metadata", {}),
            }
            comparisons.append(comparison)
            person = people.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "first_name": payload.get("first_name", ""),
                    "last_name": payload.get("last_name", ""),
                    "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
                    "score": -1.0,
                    "snapshots": payload.get("snapshots", []),
                    "metadata": payload.get("metadata", {}),
                    "point_count": 0,
                },
            )
            person["point_count"] += 1
            if score > float(person["score"]):
                person["score"] = float(score)
                person["snapshots"] = payload.get("snapshots", [])
                person["metadata"] = payload.get("metadata", {})

            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            return None

        person_matches = sorted(people.values(), key=lambda item: item["score"], reverse=True)
        best_person = person_matches[0] if person_matches else None
        second_person = person_matches[1] if len(person_matches) > 1 else None
        second_score = float(second_person["score"]) if second_person else None
        score_margin = float(best_person["score"]) - second_score if second_score is not None else None
        payload = best_record.payload or {}
        matched = bool(best_person) and float(best_person["score"]) >= threshold
        if score_margin is not None and score_margin < margin:
            matched = False

        result = {
            "person_id": best_person.get("person_id") if best_person else payload.get("person_id"),
            "first_name": best_person.get("first_name", "") if best_person else payload.get("first_name", ""),
            "last_name": best_person.get("last_name", "") if best_person else payload.get("last_name", ""),
            "full_name": best_person.get("full_name", "") if best_person else f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
            "score": round(float(best_person["score"] if best_person else best_score), 4),
            "second_score": round(second_score, 4) if second_score is not None else None,
            "score_margin": round(score_margin, 4) if score_margin is not None else None,
            "threshold": round(float(threshold), 4),
            "required_margin": round(float(margin), 4),
            "snapshots": best_person.get("snapshots", []) if best_person else payload.get("snapshots", []),
            "metadata": best_person.get("metadata", {}) if best_person else payload.get("metadata", {}),
            "matched": matched,
            "reject_reason": None if matched else (
                "below_threshold" if best_person and float(best_person["score"]) < threshold else "ambiguous_match"
            ),
            "comparisons": sorted(comparisons, key=lambda item: item["score"], reverse=True),
            "person_comparisons": [
                {
                    **item,
                    "score": round(float(item["score"]), 4),
                }
                for item in person_matches[:10]
            ],
        }
        return result

    def register(
        self,
        embedding: list[float],
        first_name: str,
        last_name: str = "",
        snapshot_path: str | None = None,
        metadata: dict | None = None,
        person_id: str | None = None,
    ) -> dict:
        normalized = self._normalize(embedding)
        self._ensure_collection(len(normalized))

        person_id = person_id or str(uuid.uuid4())
        point_id = str(uuid.uuid4())
        payload = {
            "person_id": person_id,
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "snapshots": [snapshot_path] if snapshot_path else [],
            "metadata": metadata or {},
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=normalized,
                    payload=payload,
                )
            ],
        )

        return {
            "person_id": payload["person_id"],
            "first_name": payload["first_name"],
            "last_name": payload["last_name"],
            "full_name": f'{payload["first_name"]} {payload["last_name"]}'.strip(),
            "metadata": payload["metadata"],
        }

    def add_snapshot(self, person_id: str, snapshot_path: str) -> None:
        if not snapshot_path:
            return
        if not self._collection_exists():
            return

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return

        point = records[0]
        payload = point.payload or {}
        snapshots = payload.get("snapshots", [])
        if snapshot_path in snapshots:
            return

        payload["snapshots"] = [snapshot_path]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            ],
        )

    def update_metadata(self, person_id: str, metadata: dict) -> dict | None:
        if not person_id or not self._collection_exists():
            return None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        for point in records:
            payload = point.payload or {}
            payload["metadata"] = metadata or {}
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=payload,
                    )
                ],
            )

        return self.get_person(person_id)

    def update_name(self, person_id: str, first_name: str, last_name: str = "") -> dict | None:
        if not person_id or not first_name.strip() or not self._collection_exists():
            return None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        for point in records:
            payload = point.payload or {}
            payload["first_name"] = first_name.strip()
            payload["last_name"] = last_name.strip()
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=payload,
                    )
                ],
            )

        return self.get_person(person_id)

    def get_person(self, person_id: str) -> dict | None:
        if not self._collection_exists():
            return None

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="person_id",
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not records:
            return None

        payload = records[0].payload or {}
        return {
            "person_id": payload.get("person_id"),
            "first_name": payload.get("first_name", ""),
            "last_name": payload.get("last_name", ""),
            "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
            "snapshots": payload.get("snapshots", []),
            "metadata": payload.get("metadata", {}),
        }

    def list_people(self, limit: int = 10000) -> list[dict]:
        if not self._collection_exists():
            return []

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
        )

        people = {}
        for record in records:
            payload = record.payload or {}
            person_id = payload.get("person_id") or str(record.id)
            item = people.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "first_name": payload.get("first_name", ""),
                    "last_name": payload.get("last_name", ""),
                    "full_name": f'{payload.get("first_name", "")} {payload.get("last_name", "")}'.strip(),
                    "snapshots": [],
                    "metadata": payload.get("metadata", {}),
                    "face_points": 0,
                },
            )
            item["face_points"] += 1
            for snapshot in payload.get("snapshots", []) or []:
                if snapshot and snapshot not in item["snapshots"]:
                    item["snapshots"].append(snapshot)
            if payload.get("metadata") and not item.get("metadata"):
                item["metadata"] = payload.get("metadata", {})

        return sorted(
            people.values(),
            key=lambda item: (item.get("full_name") or item.get("person_id") or "").lower(),
        )

    def reset(self) -> None:
        if self._collection_exists():
            self.client.delete_collection(self.collection_name)

    def delete_person(self, person_id: str) -> int:
        if not person_id or not self._collection_exists():
            return 0

        deleted = 0
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="person_id",
                            match=models.MatchValue(value=person_id),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
            )
            if not records:
                break
            point_ids = [record.id for record in records]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            deleted += len(point_ids)
            if offset is None:
                break
        return deleted
