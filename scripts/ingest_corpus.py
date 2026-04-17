#!/usr/bin/env python3
"""Normalize extracted drawing JSON into the canonical navigation app corpus.

Input is expected to be the JSON sidecar produced by generate_extensions_sampled_ttl.py.
Output follows schemas/application_navigation.schema.json structure.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "ttls" / "corpus_1948_2007_20py.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "navigation_corpus_1948_2007_20py.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "application_navigation.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize extracted corpus into canonical navigation schema")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input extracted drawing JSON array")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output canonical navigation corpus JSON")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Canonical schema JSON path")
    parser.add_argument("--schema-version", default="1.0.0", help="Schema version string")
    parser.add_argument("--corpus-id", default="corpus-1948-2007-max20", help="Logical corpus identifier")
    parser.add_argument("--validate-schema", action="store_true", help="Validate output using jsonschema if available")
    return parser.parse_args()


def slug_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return text or "unknown"


def label_from_qname(qname: str) -> str:
    if ":" in qname:
        local = qname.split(":", 1)[1]
    else:
        local = qname
    local = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", local)
    local = local.replace("_", " ").replace("-", " ").strip()
    local = re.sub(r"\s+", " ", local)
    return local.title() if local else qname


def normalize_dc(dc: dict[str, Any], fallback_identifier: str, fallback_title: str, fallback_source: str, fallback_year: Any) -> dict[str, Any]:
    out = {
        "dc:identifier": str(dc.get("dc:identifier") or fallback_identifier or "").strip() or None,
        "dc:title": str(dc.get("dc:title") or fallback_title or "").strip() or None,
        "dc:subject": dc.get("dc:subject") if isinstance(dc.get("dc:subject"), list) else [],
        "dc:description": str(dc.get("dc:description") or "").strip() or None,
        "dc:date": dc.get("dc:date") if dc.get("dc:date") not in (None, "") else fallback_year,
        "dc:creator": str(dc.get("dc:creator") or "").strip() or None,
        "dc:type": str(dc.get("dc:type") or "drawing").strip() or "drawing",
        "dc:format": str(dc.get("dc:format") or "image/jpeg").strip() or "image/jpeg",
        "dc:source": str(dc.get("dc:source") or fallback_source or "").strip() or None,
        "dc:relation": str(dc.get("dc:relation") or "").strip() or None,
        "dc:coverage": str(dc.get("dc:coverage") or "").strip() or None,
        "dc:rights": str(dc.get("dc:rights") or "").strip() or None,
    }
    out["dc:subject"] = [str(x).strip() for x in out["dc:subject"] if str(x).strip()]
    return out


def normalize_archdrw_drawing(arch: dict[str, Any], panel_refs: list[str]) -> dict[str, Any]:
    color_palette = arch.get("archdrw:colorPalette")
    if isinstance(color_palette, list):
        parts = [str(x).strip() for x in color_palette if str(x).strip()]
        color_val: Any = ", ".join(parts) if parts else None
    else:
        color_val = str(color_palette or "").strip() or None

    return {
        "rdf:type": arch.get("rdf:type") if isinstance(arch.get("rdf:type"), list) else [],
        "archdrw:instanceId": str(arch.get("archdrw:instanceId") or "").strip() or None,
        "archdrw:competition": str(arch.get("archdrw:competition") or "").strip() or None,
        "archdrw:layoutType": str(arch.get("archdrw:layoutType") or "").strip() or None,
        "archdrw:boardOrientation": str(arch.get("archdrw:boardOrientation") or "").strip() or None,
        "archdrw:boardPage": arch.get("archdrw:boardPage"),
        "archdrw:colorPalette": color_val,
        "archdrw:drawingStyle": str(arch.get("archdrw:drawingStyle") or "").strip() or None,
        "archdrw:medium": str(arch.get("archdrw:medium") or "").strip() or None,
        "archdrw:projection": str(arch.get("archdrw:projection") or "").strip() or None,
        "archdrw:siteContext": str(arch.get("archdrw:siteContext") or "").strip() or None,
        "archdrw:containsPanel": panel_refs,
        "archdrw:extensionTerms": [str(x).strip() for x in (arch.get("archdrw:extensionTerms") or []) if str(x).strip()],
        "archdrw:globalConceptLabels": [str(x).strip() for x in (arch.get("archdrw:globalConceptLabels") or []) if str(x).strip()],
    }


def normalize_archdrw_panel(arch: dict[str, Any]) -> dict[str, Any]:
    return {
        "rdf:type": arch.get("rdf:type") if isinstance(arch.get("rdf:type"), list) else [],
        "archdrw:panelTypeLabel": str(arch.get("archdrw:panelTypeLabel") or "").strip() or None,
        "archdrw:panelTitle": str(arch.get("archdrw:panelTitle") or "").strip() or None,
        "archdrw:panelOrder": arch.get("archdrw:panelOrder"),
        "archdrw:spatialScale": str(arch.get("archdrw:spatialScale") or "").strip() or None,
        "archdrw:hasTextTranscript": str(arch.get("archdrw:hasTextTranscript") or "").strip() or None,
        "archdrw:usesGraphicEncoding": [str(x).strip() for x in (arch.get("archdrw:usesGraphicEncoding") or []) if str(x).strip()],
        "rdfs:comment": str(arch.get("rdfs:comment") or "").strip() or None,
    }


def extract_components(drawing: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(drawing.get("panels"), list):
        return [p for p in drawing.get("panels", []) if isinstance(p, dict)]

    definitions = drawing.get("panel_definitions")
    if isinstance(definitions, dict):
        items = [v for v in definitions.values() if isinstance(v, dict)]
        return items

    return []


def build_inherited_context(drawing_norm: dict[str, Any]) -> dict[str, Any]:
    darch = drawing_norm["archdrw"]
    return {
        "drawing_ref": drawing_norm.get("drawing_ref"),
        "instance_id": drawing_norm.get("instance_id"),
        "year": drawing_norm.get("year"),
        "source_url": drawing_norm.get("source_url"),
        "dc": drawing_norm.get("dc"),
        "archdrw": {
            "archdrw:competition": darch.get("archdrw:competition"),
            "archdrw:layoutType": darch.get("archdrw:layoutType"),
            "archdrw:boardOrientation": darch.get("archdrw:boardOrientation"),
            "archdrw:globalConceptLabels": darch.get("archdrw:globalConceptLabels") or [],
        },
    }


def normalize_corpus(records: list[dict[str, Any]], schema_version: str, corpus_id: str) -> dict[str, Any]:
    drawings_out: list[dict[str, Any]] = []
    panels_out: list[dict[str, Any]] = []

    concept_map: dict[str, dict[str, Any]] = {}

    for raw in records:
        if not isinstance(raw, dict):
            continue

        instance_id = str(raw.get("instance_id") or "").strip()
        drawing_ref = str(raw.get("drawing_ref") or "").strip()
        source_url = str(raw.get("source_url") or "").strip() or None
        year_value = raw.get("year")

        dc_raw = raw.get("dc") if isinstance(raw.get("dc"), dict) else {}
        arch_raw = raw.get("archdrw") if isinstance(raw.get("archdrw"), dict) else {}

        raw_panels = extract_components(raw)
        panel_refs = [str(p.get("panel_ref") or "").strip() for p in raw_panels if str(p.get("panel_ref") or "").strip()]

        drawing_norm: dict[str, Any] = {
            "entity_type": "drawing",
            "drawing_ref": drawing_ref,
            "instance_id": instance_id,
            "year": year_value,
            "source_url": source_url,
            "dc": normalize_dc(
                dc_raw,
                fallback_identifier=instance_id,
                fallback_title=instance_id,
                fallback_source=source_url or "",
                fallback_year=year_value,
            ),
            "archdrw": normalize_archdrw_drawing(arch_raw, panel_refs),
            "concept_refs": [str(x).strip() for x in (raw.get("concept_refs") or []) if str(x).strip()],
            "panels": [],
        }

        inherited = build_inherited_context(drawing_norm)

        for idx, panel_raw in enumerate(raw_panels, start=1):
            parch_raw = panel_raw.get("archdrw") if isinstance(panel_raw.get("archdrw"), dict) else {}
            panel_type_qname = str(panel_raw.get("panel_type_qname") or "archdrw:Panel").strip() or "archdrw:Panel"
            panel_ref = str(panel_raw.get("panel_ref") or "").strip()
            panel_id = str(panel_raw.get("panel_id") or f"p{idx}").strip()
            panel_type_label = str(panel_raw.get("panel_type") or panel_raw.get("panel_type_label") or parch_raw.get("archdrw:panelTypeLabel") or "Panel").strip() or "Panel"

            concept_refs = [str(x).strip() for x in (panel_raw.get("concept_refs") or []) if str(x).strip()]
            if not concept_refs:
                concept_refs = [str(x).strip() for x in (panel_raw.get("extracted_terms") or []) if str(x).strip() and ":" in str(x)]

            panel_norm = {
                "entity_type": "panel",
                "panel_ref": panel_ref,
                "panel_id": panel_id,
                "year": panel_raw.get("year") if panel_raw.get("year") is not None else year_value,
                "panel_type_qname": panel_type_qname,
                "panel_title": str(panel_raw.get("panel_title") or "").strip() or None,
                "panel_order": panel_raw.get("panel_order"),
                "bbox": str(panel_raw.get("bbox") or "").strip() or None,
                "ocr_text": str(panel_raw.get("ocr_text") or "").strip() or None,
                "local_description": str(panel_raw.get("local_description") or "").strip() or None,
                "extracted_terms": [str(x).strip() for x in (panel_raw.get("extracted_terms") or []) if str(x).strip()],
                "archdrw": normalize_archdrw_panel(parch_raw),
                "inherited_context": inherited,
                "concept_refs": concept_refs,
            }

            drawing_norm["panels"].append(panel_norm)
            panels_out.append(panel_norm)

            for cref in concept_refs:
                if ":" not in cref:
                    continue
                concept_entry = concept_map.setdefault(
                    cref,
                    {
                        "entity_type": "concept",
                        "concept_ref": cref,
                        "label": label_from_qname(cref),
                        "qname": cref,
                        "concept_type": "ontology",
                        "description": None,
                        "broader_refs": [],
                        "related_refs": [],
                        "supporting_drawing_refs": set(),
                        "supporting_panel_refs": set(),
                    },
                )
                concept_entry["supporting_drawing_refs"].add(drawing_ref)
                concept_entry["supporting_panel_refs"].add(panel_ref)

        for cref in drawing_norm["concept_refs"]:
            if ":" not in cref:
                continue
            concept_entry = concept_map.setdefault(
                cref,
                {
                    "entity_type": "concept",
                    "concept_ref": cref,
                    "label": label_from_qname(cref),
                    "qname": cref,
                    "concept_type": "ontology",
                    "description": None,
                    "broader_refs": [],
                    "related_refs": [],
                    "supporting_drawing_refs": set(),
                    "supporting_panel_refs": set(),
                },
            )
            concept_entry["supporting_drawing_refs"].add(drawing_ref)

        drawings_out.append(drawing_norm)

    concepts_out = []
    for concept_ref in sorted(concept_map.keys()):
        c = concept_map[concept_ref]
        concepts_out.append(
            {
                "entity_type": "concept",
                "concept_ref": c["concept_ref"],
                "label": c["label"],
                "qname": c["qname"],
                "concept_type": c["concept_type"],
                "description": c["description"],
                "broader_refs": c["broader_refs"],
                "related_refs": c["related_refs"],
                "supporting_drawing_refs": sorted([x for x in c["supporting_drawing_refs"] if x]),
                "supporting_panel_refs": sorted([x for x in c["supporting_panel_refs"] if x]),
            }
        )

    return {
        "schema_version": schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus_id,
        "drawings": drawings_out,
        "panels": panels_out,
        "concepts": concepts_out,
        "sessions": [],
    }


def validate_with_jsonschema(payload: dict[str, Any], schema_path: Path) -> tuple[bool, str]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return False, "jsonschema package not installed; skipped validation"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except Exception as exc:  # noqa: BLE001
        return False, f"validation failed: {exc}"

    return True, "validation passed"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    schema_path = Path(args.schema)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not schema_path.exists():
        raise SystemExit(f"Schema file not found: {schema_path}")

    raw_data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise SystemExit("Input JSON must be an array of drawing entries")

    normalized = normalize_corpus(raw_data, schema_version=args.schema_version, corpus_id=args.corpus_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    print(f"Input drawings: {len(raw_data)}")
    print(f"Output drawings: {len(normalized['drawings'])}")
    print(f"Output panels: {len(normalized['panels'])}")
    print(f"Output concepts: {len(normalized['concepts'])}")
    print(f"Written -> {output_path}")

    if args.validate_schema:
        ok, msg = validate_with_jsonschema(normalized, schema_path)
        print(f"Schema validation: {msg}")
        if not ok and msg.startswith("validation failed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
