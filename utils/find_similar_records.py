#!/usr/bin/env python3
"""Find similar archival image records and summarize group profile fields.

This utility builds a lightweight TF-IDF vector space from metadata fields and uses
cosine similarity to retrieve nearest neighbors. It can also summarize which
Dublin Core and archdrw parameters/values are most prevalent in a selected group.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class VectorSpace:
    records: list[dict[str, Any]]
    features: list[dict[int, float]]
    norms: list[float]
    index_by_instance: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find nearest neighbors for image records using cosine similarity, "
            "and summarize group prevalence for dc/archdrw fields."
        )
    )
    parser.add_argument(
        "--input",
        default="frontend/public/data/enriched_metadata.json",
        help="Input metadata JSON (default: frontend/public/data/enriched_metadata.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_neighbors = subparsers.add_parser("neighbors", help="Find nearest neighbors")
    p_neighbors.add_argument("--instance-id", required=True, help="Query instance_id")
    p_neighbors.add_argument("--top-k", type=int, default=10, help="Neighbors to return")
    p_neighbors.add_argument(
        "--include-self",
        action="store_true",
        help="Include the query record in results",
    )

    p_group = subparsers.add_parser("group-profile", help="Summarize a record group")
    p_group.add_argument(
        "--instance-ids",
        default="",
        help="Comma-separated instance IDs for explicit group selection",
    )
    p_group.add_argument(
        "--title",
        default="",
        help="Use all records with exact matching title",
    )
    p_group.add_argument(
        "--top-params",
        type=int,
        default=20,
        help="Top parameter names by prevalence",
    )
    p_group.add_argument(
        "--top-values",
        type=int,
        default=30,
        help="Top parameter values by prevalence",
    )

    p_both = subparsers.add_parser(
        "neighbors-profile",
        help="Find neighbors, then profile query+neighbors as a group",
    )
    p_both.add_argument("--instance-id", required=True, help="Query instance_id")
    p_both.add_argument("--top-k", type=int, default=10, help="Neighbors to include")
    p_both.add_argument(
        "--top-params",
        type=int,
        default=20,
        help="Top parameter names by prevalence",
    )
    p_both.add_argument(
        "--top-values",
        type=int,
        default=30,
        help="Top parameter values by prevalence",
    )

    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def add_feature(features: list[str], key: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, list):
        for item in value:
            add_feature(features, key, item)
        return

    value_text = str(value).strip()
    if not value_text:
        return

    normalized = value_text.lower().replace(" ", "_")
    features.append(f"{key}={normalized}")

    # Also include token-level terms for semantic overlap across fields.
    for token in tokenize(value_text):
        features.append(f"tok:{token}")


def record_features(record: dict[str, Any]) -> list[str]:
    feats: list[str] = []

    # Core identifiers/context
    add_feature(feats, "title", record.get("title"))
    add_feature(feats, "project_key", record.get("project_key"))
    add_feature(feats, "year", record.get("year"))

    # Dublin Core
    dc = record.get("dublin_core")
    if isinstance(dc, dict):
        for key in sorted(dc.keys()):
            add_feature(feats, key, dc.get(key))

    # Drawing ontology fields
    archdrw = record.get("archdrw")
    if isinstance(archdrw, dict):
        for key in sorted(archdrw.keys()):
            add_feature(feats, f"archdrw:{key}", archdrw.get(key))

    # Tags (if present)
    tags = record.get("tags")
    if isinstance(tags, list):
        add_feature(feats, "tags", tags)

    return feats


def build_vector_space(records: list[dict[str, Any]]) -> VectorSpace:
    tokenized_docs: list[list[str]] = []
    for record in records:
        tokenized_docs.append(record_features(record))

    vocab: dict[str, int] = {}
    df = collections.Counter()

    for tokens in tokenized_docs:
        seen: set[str] = set()
        for term in tokens:
            if term not in vocab:
                vocab[term] = len(vocab)
            if term not in seen:
                df[term] += 1
                seen.add(term)

    n_docs = len(records)
    idf: list[float] = [0.0] * len(vocab)
    for term, idx in vocab.items():
        # Smoothed IDF keeps values finite and robust for small groups.
        idf[idx] = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0

    features: list[dict[int, float]] = []
    norms: list[float] = []

    for tokens in tokenized_docs:
        tf = collections.Counter(tokens)
        vec: dict[int, float] = {}
        for term, count in tf.items():
            idx = vocab[term]
            weight = (1.0 + math.log(count)) * idf[idx]
            vec[idx] = weight

        norm = math.sqrt(sum(v * v for v in vec.values()))
        features.append(vec)
        norms.append(norm)

    index_by_instance: dict[str, int] = {}
    for i, rec in enumerate(records):
        instance_id = str(rec.get("instance_id", "")).strip()
        if instance_id:
            index_by_instance[instance_id] = i

    return VectorSpace(
        records=records,
        features=features,
        norms=norms,
        index_by_instance=index_by_instance,
    )


def cosine(vec_a: dict[int, float], norm_a: float, vec_b: dict[int, float], norm_b: float) -> float:
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a

    dot = 0.0
    for idx, val in vec_a.items():
        other = vec_b.get(idx)
        if other is not None:
            dot += val * other

    return dot / (norm_a * norm_b)


def nearest_neighbors(space: VectorSpace, instance_id: str, top_k: int, include_self: bool) -> list[tuple[int, float]]:
    query_idx = space.index_by_instance.get(instance_id)
    if query_idx is None:
        raise SystemExit(f"instance_id not found: {instance_id}")

    query_vec = space.features[query_idx]
    query_norm = space.norms[query_idx]

    scores: list[tuple[int, float]] = []
    for i, vec in enumerate(space.features):
        if not include_self and i == query_idx:
            continue
        score = cosine(query_vec, query_norm, vec, space.norms[i])
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[: max(0, top_k)]


def normalize_scalar(value: Any) -> str:
    return str(value).strip().lower()


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    text = normalize_scalar(value)
    return [text] if text else []


def summarize_group(records: list[dict[str, Any]], top_params: int, top_values: int) -> dict[str, Any]:
    parameter_presence = collections.Counter()
    parameter_values = collections.Counter()

    for rec in records:
        dc = rec.get("dublin_core")
        if isinstance(dc, dict):
            for key, value in dc.items():
                values = flatten_values(value)
                if values:
                    parameter_presence[f"dc::{key}"] += 1
                    for v in values:
                        parameter_values[f"dc::{key}={v}"] += 1

        arch = rec.get("archdrw")
        if isinstance(arch, dict):
            for key, value in arch.items():
                values = flatten_values(value)
                if values:
                    parameter_presence[f"archdrw::{key}"] += 1
                    for v in values:
                        parameter_values[f"archdrw::{key}={v}"] += 1

    return {
        "group_size": len(records),
        "top_parameters": parameter_presence.most_common(max(0, top_params)),
        "top_parameter_values": parameter_values.most_common(max(0, top_values)),
    }


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be an array of records")

    return [rec for rec in data if isinstance(rec, dict)]


def resolve_group_records(all_records: list[dict[str, Any]], instance_ids: str, title: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rec in all_records:
        iid = str(rec.get("instance_id", "")).strip()
        if iid:
            by_id[iid] = rec

    if instance_ids.strip():
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for iid in [x.strip() for x in instance_ids.split(",") if x.strip()]:
            rec = by_id.get(iid)
            if rec is None:
                missing.append(iid)
            else:
                selected.append(rec)
        if missing:
            raise SystemExit(f"instance_id(s) not found: {', '.join(missing)}")
        return selected

    if title.strip():
        return [rec for rec in all_records if str(rec.get("title", "")).strip() == title]

    raise SystemExit("Provide either --instance-ids or --title for group-profile")


def print_neighbors(space: VectorSpace, query_id: str, neighbors: list[tuple[int, float]]) -> None:
    print(f"QUERY\t{query_id}")
    print("rank\tscore\tinstance_id\ttitle\tpage\tyear")
    for rank, (idx, score) in enumerate(neighbors, start=1):
        rec = space.records[idx]
        print(
            f"{rank}\t{score:.6f}\t{rec.get('instance_id', '')}\t"
            f"{rec.get('title', '')}\t{rec.get('page', '')}\t{rec.get('year', '')}"
        )


def print_group_summary(summary: dict[str, Any]) -> None:
    print(f"GROUP_SIZE\t{summary['group_size']}")
    print("TOP_PARAMETERS")
    for key, count in summary["top_parameters"]:
        print(f"{key}\t{count}")

    print("TOP_PARAMETER_VALUES")
    for key, count in summary["top_parameter_values"]:
        print(f"{key}\t{count}")


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input))

    if args.command == "neighbors":
        space = build_vector_space(records)
        nn = nearest_neighbors(
            space=space,
            instance_id=args.instance_id,
            top_k=args.top_k,
            include_self=args.include_self,
        )
        print_neighbors(space, args.instance_id, nn)
        return

    if args.command == "group-profile":
        group = resolve_group_records(records, args.instance_ids, args.title)
        summary = summarize_group(group, args.top_params, args.top_values)
        print_group_summary(summary)
        return

    if args.command == "neighbors-profile":
        space = build_vector_space(records)
        nn = nearest_neighbors(
            space=space,
            instance_id=args.instance_id,
            top_k=args.top_k,
            include_self=False,
        )
        print_neighbors(space, args.instance_id, nn)

        query_idx = space.index_by_instance[args.instance_id]
        group_records = [space.records[query_idx]] + [space.records[idx] for idx, _ in nn]
        summary = summarize_group(group_records, args.top_params, args.top_values)
        print("GROUP_PROFILE_FOR_QUERY_PLUS_NEIGHBORS")
        print_group_summary(summary)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
