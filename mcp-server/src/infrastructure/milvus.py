"""
Milvus Vector Database Adapter — Hybrid Search (Dense COSINE + BM25)

컬렉션: rnd_nodes
  id:        VARCHAR(64)           — 노드 ID (PK)
  node_type: VARCHAR(32)           — Paper / Patent / Technology / Researcher / Project
  year:      INT64                 — 연도 (필터용)
  name:      VARCHAR(512)          — 제목/이름
  text:      VARCHAR(65535)        — BM25 전문검색 원본
  sparse:    SPARSE_FLOAT_VECTOR   — BM25 자동 생성
  dense:     FLOAT_VECTOR(768)     — KR-SBERT 임베딩
"""

from __future__ import annotations
from typing import Any, Callable
import logging
import time

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rnd_nodes"
VECTOR_DIM = 768


def make_vector_search_fn(
    client: Any,
    embedding_fn: Callable,
    collection_name: str = COLLECTION_NAME,
) -> Callable:
    """
    Milvus hybrid_search 콜백 생성.
    인터페이스: (query, node_type, top_k, dense_weight, sparse_weight) → list[tuple[str, float]]
    """
    def vector_search(
        query: str,
        node_type: str = "",
        top_k: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        year_from: int = 0,
        year_to: int = 0,
    ) -> list[tuple[str, float, str, str]]:
        from pymilvus import AnnSearchRequest, WeightedRanker

        t_embed = time.perf_counter()
        vector = embedding_fn([query])[0].tolist()

        filter_parts: list[str] = []
        if node_type:
            filter_parts.append(f'node_type == "{node_type}"')
        if year_from:
            filter_parts.append(f"year >= {year_from}")
        if year_to:
            filter_parts.append(f"year <= {year_to}")
        expr = " && ".join(filter_parts)

        search_limit = top_k * 2

        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=search_limit,
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=search_limit,
            expr=expr or None,
        )

        t_search = time.perf_counter()
        try:
            results = client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(dense_weight, sparse_weight),
                limit=top_k,
                output_fields=["id", "node_type", "name"],
            )
        except Exception as e:
            logger.warning("[Milvus] hybrid_search 실패, dense fallback: %s", e)
            results = client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field="dense",
                limit=top_k,
                filter=expr or None,
                output_fields=["id", "node_type", "name"],
                search_params={"metric_type": "COSINE", "params": {}},
            )

        search_ms = (time.perf_counter() - t_search) * 1000
        embed_ms  = (t_search - t_embed) * 1000

        if not results or not results[0]:
            logger.debug("[Milvus] '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=0",
                         query, node_type or "*", embed_ms, search_ms)
            return []

        hits = results[0]
        logger.debug("[Milvus] '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=%d",
                     query, node_type or "*", embed_ms, search_ms, len(hits))
        return [
            (hit["entity"]["id"], float(hit["distance"]), hit["entity"].get("node_type", ""), hit["entity"].get("name", ""))
            for hit in hits
        ]

    return vector_search


def ensure_collection(client: Any, collection_name: str = COLLECTION_NAME) -> None:
    """rnd_nodes 컬렉션 없으면 생성 (Dense HNSW/COSINE + Sparse BM25)."""
    from pymilvus import DataType, Function, FunctionType

    if client.has_collection(collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",        DataType.VARCHAR, max_length=64,  is_primary=True)
    schema.add_field("node_type", DataType.VARCHAR, max_length=32)
    schema.add_field("name",      DataType.VARCHAR, max_length=512)
    schema.add_field("year",      DataType.INT64)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=65535,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "standard"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense",  DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    schema.add_function(Function(
        name="text_to_sparse",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("[Milvus] 컬렉션 '%s' 생성 완료 (Dense COSINE + BM25 hybrid)", collection_name)
