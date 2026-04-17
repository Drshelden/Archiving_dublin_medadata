#!/usr/bin/env python3
"""Generate sampled RDF instances using OpenAI review + merged extension ontology.

Features:
- Loads metadata records from JSON.
- Limits records per title (default max 10 per title).
- Calls OpenAI Vision for detailed per-instance review.
- Populates Dublin Core and extension-linked instance triples.
- Writes Turtle output to data/ttls/extensions_sampled.ttl by default.
"""

from __future__ import annotations

import argparse
import base64
import json
import io
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

try:
    import requests
except ImportError:
    print("ERROR: requests package not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    cv2 = None
    np = None
    Image = None

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "frontend" / "public" / "data" / "enriched_metadata.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "archdrw.ttl"
DEFAULT_OUTPUT = ROOT / "data" / "ttls" / "extensions_sampled.ttl"
DEFAULT_BOARDS_ROOT = ROOT / "data" / "boards"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

ARCHIVE_DATA = "https://archivingresearch.org/data/"
ARCHDRW_NS = "https://archivingresearch.org/schema/drawing#"
ARCHGEN_NS = "https://archivingresearch.org/schema/drawing/extensions/generated#"
DC_NS = "http://purl.org/dc/elements/1.1/"


def load_local_env() -> None:
    for env_path in (ROOT / ".env", ROOT / "backend" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sampled extension RDF using OpenAI")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input metadata JSON")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Merged extension ontology TTL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output Turtle file")
    parser.add_argument("--json-output", default="", help="Optional JSON sidecar output for structured board data")
    parser.add_argument("--boards-root", default=str(DEFAULT_BOARDS_ROOT), help="Local board image root organized by year folders")
    parser.add_argument("--min-year", type=int, default=1948, help="Minimum year folder to include")
    parser.add_argument("--max-year", type=int, default=2007, help="Maximum year folder to include")
    parser.add_argument("--max-per-year", type=int, default=10, help="Max records sampled from any included year")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model")
    parser.add_argument("--max-per-title", type=int, default=10, help="Max records per title")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--max-instances", type=int, default=0, help="Optional overall cap after title sampling")
    parser.add_argument("--no-openai", action="store_true", help="Skip OpenAI calls and use existing fields only")
    parser.add_argument("--log-every", type=int, default=20, help="Progress print interval")
    parser.add_argument(
        "--panel-data-mode",
        choices=("references", "embedded"),
        default="references",
        help="How panel data is represented in the JSON sidecar",
    )
    return parser.parse_args()


def ttl_esc(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def lit(value: str) -> str:
    return f'"{ttl_esc(str(value))}"'


def lit_int(value: Any) -> str:
    try:
        return f'"{int(value)}"^^xsd:integer'
    except (TypeError, ValueError):
        return lit(str(value))


def draw_iri(instance_id: str) -> str:
    return f"<{ARCHIVE_DATA}{quote(instance_id, safe='-._~')}>"


def normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_schema_terms(schema_text: str) -> dict[str, str]:
    """Parse referenceable archdrw/archgen terms from ontology TTL.

    Includes classes and controlled instances, so extracted terms can map to
    both `archgen:*` and richer `archdrw:*` vocabulary.
    """
    mapping: dict[str, str] = {}

    pattern_class = re.compile(
        r"(archdrw|archgen):([A-Za-z0-9_]+)\s+\n?\s*a\s+owl:Class",
        flags=re.MULTILINE,
    )
    pattern_controlled = re.compile(
        r"(archdrw|archgen):([A-Za-z0-9_]+)\s+\n?\s*a\s+(archdrw:[A-Za-z0-9_]+|archgen:[A-Za-z0-9_]+)",
        flags=re.MULTILINE,
    )

    for prefix, local in pattern_class.findall(schema_text):
        qname = f"{prefix}:{local}"
        mapping[normalize_token(local)] = qname

    for prefix, local, _rtype in pattern_controlled.findall(schema_text):
        qname = f"{prefix}:{local}"
        mapping.setdefault(normalize_token(local), qname)

    return mapping


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def extract_record_filename(rec: dict[str, Any]) -> str:
    url = str(rec.get("url") or "").strip()
    if url:
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        if name:
            return name

    instance_id = str(rec.get("instance_id") or "").strip()
    if instance_id:
        return f"{instance_id}.jpg"
    return ""


def load_board_index(boards_root: Path, min_year: int, max_year: int) -> tuple[dict[str, dict[str, Any]], dict[int, int]]:
    index: dict[str, dict[str, Any]] = {}
    counts: dict[int, int] = defaultdict(int)
    if not boards_root.exists():
        raise SystemExit(f"Boards root not found: {boards_root}")

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    for year_dir in sorted(boards_root.iterdir(), key=lambda path: path.name):
        if not year_dir.is_dir():
            continue
        try:
            year_value = int(year_dir.name)
        except ValueError:
            continue
        if year_value < min_year or year_value > max_year:
            continue

        for image_path in sorted(year_dir.iterdir(), key=lambda path: path.name):
            if not image_path.is_file() or image_path.suffix.lower() not in allowed_suffixes:
                continue
            index[image_path.name] = {
                "year": year_value,
                "local_image_path": str(image_path),
            }
            counts[year_value] += 1

    return index, counts


def attach_local_board_metadata(records: list[dict[str, Any]], board_index: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    matched: list[dict[str, Any]] = []
    unmatched = 0
    for rec in records:
        filename = extract_record_filename(rec)
        board_info = board_index.get(filename)
        if not board_info:
            unmatched += 1
            continue

        enriched = dict(rec)
        enriched["year"] = board_info["year"]
        enriched["local_image_path"] = board_info["local_image_path"]
        matched.append(enriched)

    return matched, unmatched


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    left_area = max(1, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    return inter / float(left_area + right_area - inter)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    return x1, y1, x2, y2


def normalize_box(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return f"{x1 / width:.3f},{y1 / height:.3f},{x2 / width:.3f},{y2 / height:.3f}"


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_region_text(np_img: Any, box: tuple[int, int, int, int]) -> str:
    if pytesseract is None or Image is None:
        return ""
    x1, y1, x2, y2 = box
    crop = np_img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    try:
        text = pytesseract.image_to_string(Image.fromarray(gray), config="--psm 6", timeout=8)
    except Exception:  # noqa: BLE001
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 4:
        return ""
    return text[:220]


def extract_global_ocr_blocks(np_img: Any) -> list[str]:
    if pytesseract is None or Image is None:
        return []
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(gray),
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        return []
    blocks: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx, raw_text in enumerate(data.get("text", [])):
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if len(text) < 3:
            continue
        try:
            conf = float(data["conf"][idx])
        except (TypeError, ValueError):
            continue
        if conf < 35:
            continue
        key = (int(data["block_num"][idx]), int(data["par_num"][idx]), int(data["line_num"][idx]))
        entry = blocks.setdefault(
            key,
            {"parts": [], "left": [], "top": [], "right": [], "bottom": [], "conf": []},
        )
        left = int(data["left"][idx])
        top = int(data["top"][idx])
        width = int(data["width"][idx])
        height = int(data["height"][idx])
        entry["parts"].append(text)
        entry["left"].append(left)
        entry["top"].append(top)
        entry["right"].append(left + width)
        entry["bottom"].append(top + height)
        entry["conf"].append(conf)

    summaries: list[tuple[float, str]] = []
    for entry in blocks.values():
        text = re.sub(r"\s+", " ", " ".join(entry["parts"]).strip())
        if len(text) < 8:
            continue
        avg_conf = sum(entry["conf"]) / len(entry["conf"])
        x1 = min(entry["left"])
        y1 = min(entry["top"])
        x2 = max(entry["right"])
        y2 = max(entry["bottom"])
        summaries.append((avg_conf, f"ocr[{x1},{y1},{x2},{y2}] {text[:180]}"))

    summaries.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in summaries[:8]]


def build_image_region_hints(image_url: str, local_image_path: str = "") -> dict[str, Any]:
    empty = {"region_summary": "", "crop_urls": [], "ocr_lines": []}
    if cv2 is None or np is None or Image is None:
        return empty

    try:
        if local_image_path:
            image = Image.open(local_image_path).convert("RGB")
        else:
            if not image_url:
                return empty
            response = requests.get(image_url, timeout=45)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:  # noqa: BLE001
        return empty

    np_img = np.array(image)
    height, width = np_img.shape[:2]
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals: list[dict[str, Any]] = []
    image_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / image_area
        if area_ratio < 0.025 or area_ratio > 0.72:
            continue
        if w < width * 0.14 or h < height * 0.1:
            continue
        proposals.append(
            {
                "box": clamp_box((x, y, x + w, y + h), width, height, pad=18),
                "score": area_ratio,
                "source": "contour",
            }
        )

    half_w = width // 2
    half_h = height // 2
    fallback_boxes = [
        (0, 0, half_w, half_h),
        (half_w, 0, width, half_h),
        (0, half_h, half_w, height),
        (half_w, half_h, width, height),
    ]
    for box in fallback_boxes:
        proposals.append({"box": clamp_box(box, width, height, pad=0), "score": 0.12, "source": "grid"})

    proposals.sort(key=lambda item: item["score"], reverse=True)

    kept: list[dict[str, Any]] = []
    for proposal in proposals:
        if any(bbox_iou(proposal["box"], existing["box"]) > 0.6 for existing in kept):
            continue
        kept.append(proposal)
        if len(kept) >= 6:
            break

    crop_urls: list[str] = []
    region_lines: list[str] = []
    for idx, proposal in enumerate(sorted(kept, key=lambda item: (item["box"][1], item["box"][0])), start=1):
        box = proposal["box"]
        x1, y1, x2, y2 = box
        crop = image.crop(box)
        longest_side = max(crop.size)
        if longest_side > 1200:
            scale = 1200 / float(longest_side)
            crop = crop.resize((max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale))))
        crop_urls.append(image_to_data_url(crop))
        region_text = extract_region_text(np_img, box)
        source = proposal["source"]
        text_clause = f" text={region_text}" if region_text else ""
        region_lines.append(
            f"region_{idx} source={source} bbox={normalize_box(box, width, height)}{text_clause}"
        )

    return {
        "region_summary": "\n".join(region_lines),
        "crop_urls": crop_urls,
        "ocr_lines": extract_global_ocr_blocks(np_img),
    }


def call_openai_review(api_key: str, model: str, rec: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(rec.get("instance_id") or "")
    title = str(rec.get("title") or "")
    year = rec.get("year")
    project = str(rec.get("project_key") or "")
    url = str(rec.get("url") or "")
    local_image_path = str(rec.get("local_image_path") or "").strip()
    image_hints = build_image_region_hints(url, local_image_path)
    ocr_summary = "\n".join(image_hints.get("ocr_lines") or [])
    region_summary = str(image_hints.get("region_summary") or "").strip()

    prompt = (
        "Review this architectural board and return JSON only. "
        "Use this exact structure: "
        "{"
        "dc_subject: string[], "
        "dc_description: string, "
        "extension_terms: string[], "
        "board_layout: {layout_type: string, board_orientation: string}, "
        "components: [{panel_id: string, panel_type: string, panel_title: string, panel_order: integer, "
        "bbox: string, ocr_text: string, local_description: string, extracted_terms: string[]}], "
        "global_concepts: string[]"
        "}. "
        "For components, identify meaningful sub-drawings (plans, sections, details, site map, text block, renderings). "
        "A drawing may contain zero panels, exactly one panel, or many panels; if the image is effectively a single drawing field, "
        "return one component rather than forcing an artificial subdivision. "
        "Candidate subregions and OCR hints may be provided alongside the full board; use them to recover dense layouts, "
        "but merge overlapping hints into meaningful archival components rather than repeating them literally. "
        "Base response on visible content and metadata context only; do not invent facts."
    )

    user_text = (
        f"instance_id: {instance_id}\n"
        f"title: {title}\n"
        f"year: {year}\n"
        f"project_key: {project}\n"
    )
    if region_summary:
        user_text += f"\nCandidate regions:\n{region_summary}\n"
    if ocr_summary:
        user_text += f"\nOCR hints:\n{ocr_summary}\n"
    user_text += f"\n{prompt}"

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": user_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": "high",
            },
        },
    ]
    for crop_url in image_hints.get("crop_urls") or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": crop_url,
                    "detail": "low",
                },
            }
        )

    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are an expert architectural archivist producing high-quality structured metadata.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=body, timeout=90)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = parse_json_object(content)
    if not parsed:
        return {
            "dc_subject": [],
            "dc_description": "",
            "extension_terms": [],
            "board_layout": {},
            "components": [],
            "global_concepts": [],
        }
    return parsed


def sample_by_title(records: list[dict[str, Any]], max_per_title: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        title = str(rec.get("title") or "").strip() or "(untitled)"
        groups[title].append(rec)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for title in sorted(groups.keys()):
        items = groups[title]
        if len(items) <= max_per_title:
            sampled.extend(items)
        else:
            sampled.extend(rng.sample(items, max_per_title))
    return sampled


def sample_by_year(records: list[dict[str, Any]], max_per_year: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        try:
            year_value = int(rec.get("year"))
        except (TypeError, ValueError):
            continue
        groups[year_value].append(rec)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for year_value in sorted(groups.keys()):
        items = sorted(groups[year_value], key=lambda item: str(item.get("instance_id") or ""))
        if len(items) <= max_per_year:
            selected = items
        else:
            selected = sorted(rng.sample(items, max_per_year), key=lambda item: str(item.get("instance_id") or ""))
        sampled.extend(selected)
    return sampled


def component_iri(instance_id: str, panel_id: str) -> str:
    safe_panel = quote(panel_id.strip() or "panel", safe="-._~")
    return f"<{ARCHIVE_DATA}{quote(instance_id, safe='-._~')}/panel/{safe_panel}>"


def normalize_component(
    comp: dict[str, Any],
    idx: int,
    instance_id: str,
    term_map: dict[str, str],
    drawing_year: Any,
) -> dict[str, Any]:
    panel_id = str(comp.get("panel_id") or f"p{idx}").strip()
    panel_ref = component_iri(instance_id, panel_id)
    panel_type_text = str(comp.get("panel_type") or "").strip()
    mapped_panel_type = term_map.get(normalize_token(panel_type_text)) if panel_type_text else None
    panel_type_qname = mapped_panel_type or "archdrw:Panel"

    extracted_terms_qnames: list[str] = []
    extracted_terms = comp.get("extracted_terms")
    if isinstance(extracted_terms, list):
        for term in extracted_terms:
            qname = term_map.get(normalize_token(str(term)))
            if qname and qname not in extracted_terms_qnames:
                extracted_terms_qnames.append(qname)

    return {
        "entity_type": "panel",
        "panel_id": panel_id,
        "panel_ref": panel_ref.strip("<>"),
        "year": drawing_year,
        "panel_type_qname": panel_type_qname,
        "panel_title": str(comp.get("panel_title") or "").strip(),
        "panel_order": comp.get("panel_order"),
        "bbox": str(comp.get("bbox") or "").strip(),
        "ocr_text": str(comp.get("ocr_text") or "").strip(),
        "local_description": str(comp.get("local_description") or "").strip(),
        "extracted_terms": extracted_terms_qnames,
        "archdrw": {
            "rdf:type": ["archdrw:BoardComponent", panel_type_qname],
            "archdrw:panelTypeLabel": panel_type_text or "Panel",
            "archdrw:panelTitle": str(comp.get("panel_title") or "").strip(),
            "archdrw:panelOrder": comp.get("panel_order"),
            "archdrw:spatialScale": str(comp.get("bbox") or "").strip(),
            "archdrw:hasTextTranscript": str(comp.get("ocr_text") or "").strip(),
            "rdfs:comment": str(comp.get("local_description") or "").strip(),
            "archdrw:usesGraphicEncoding": extracted_terms_qnames,
        },
    }


def build_drawing_json_entry(
    rec: dict[str, Any],
    review: dict[str, Any],
    term_map: dict[str, str],
    panel_data_mode: str,
) -> dict[str, Any]:
    iid = str(rec.get("instance_id") or "").strip()
    drawing_year = rec.get("year")
    dc_map = dict(rec.get("dublin_core") or {})
    arch_map = dict(rec.get("archdrw") or {})

    dc_subject = review.get("dc_subject") if isinstance(review.get("dc_subject"), list) else []
    dc_description = str(review.get("dc_description") or dc_map.get("dc:description") or "").strip()
    board_layout = review.get("board_layout") if isinstance(review.get("board_layout"), dict) else {}
    layout_type = str(board_layout.get("layout_type") or "").strip()
    board_orientation = str(board_layout.get("board_orientation") or "").strip()
    extension_terms = review.get("extension_terms") if isinstance(review.get("extension_terms"), list) else []
    global_concepts = review.get("global_concepts") if isinstance(review.get("global_concepts"), list) else []

    entry: dict[str, Any] = {
        "entity_type": "drawing",
        "instance_id": iid,
        "drawing_ref": draw_iri(iid).strip("<>"),
        "year": drawing_year,
        "source_url": str(rec.get("url") or dc_map.get("dc:source") or "").strip(),
        "dc": {
            "dc:identifier": iid,
            "dc:title": str(dc_map.get("dc:title") or rec.get("title") or "").strip(),
            "dc:creator": str(dc_map.get("dc:creator") or "").strip(),
            "dc:subject": dc_subject,
            "dc:description": dc_description,
            "dc:date": dc_map.get("dc:date") if dc_map.get("dc:date") not in (None, "") else drawing_year,
            "dc:type": str(dc_map.get("dc:type") or rec.get("type") or "drawing").strip() or "drawing",
            "dc:format": str(dc_map.get("dc:format") or "image/jpeg").strip() or "image/jpeg",
            "dc:source": str(rec.get("url") or dc_map.get("dc:source") or "").strip(),
            "dc:relation": str(dc_map.get("dc:relation") or rec.get("project_key") or "").strip(),
            "dc:coverage": str(dc_map.get("dc:coverage") or "").strip(),
            "dc:rights": str(dc_map.get("dc:rights") or "").strip(),
        },
        "archdrw": {
            "rdf:type": ["archdrw:ArchivalDrawing"] + (["archdrw:PresentationBoard"] if isinstance(review.get("components"), list) and review.get("components") else []),
            "archdrw:instanceId": iid,
            "archdrw:competition": str(dc_map.get("dc:relation") or rec.get("project_key") or "").strip(),
            "archdrw:boardPage": rec.get("page"),
            "archdrw:layoutType": layout_type,
            "archdrw:boardOrientation": board_orientation,
            "archdrw:medium": str(arch_map.get("medium") or "").strip(),
            "archdrw:drawingStyle": str(arch_map.get("drawingStyle") or "").strip(),
            "archdrw:siteContext": str(arch_map.get("siteContext") or "").strip(),
            "archdrw:projection": str(arch_map.get("projection") or "").strip(),
            "archdrw:colorPalette": [str(color).strip() for color in (arch_map.get("colorPalette") or []) if str(color).strip()],
            "archdrw:extensionTerms": [str(x).strip() for x in extension_terms if str(x).strip()],
            "archdrw:globalConceptLabels": [str(x).strip() for x in global_concepts if str(x).strip()],
        },
    }

    raw_components = review.get("components")
    components = [
        normalize_component(comp, idx, iid, term_map, drawing_year)
        for idx, comp in enumerate(raw_components, start=1)
        if isinstance(comp, dict)
    ] if isinstance(raw_components, list) else []

    entry["archdrw"]["archdrw:containsPanel"] = [comp["panel_ref"] for comp in components]

    if panel_data_mode == "embedded":
        entry["panels"] = components
    else:
        entry["panel_refs"] = [comp["panel_ref"] for comp in components]
        entry["panel_definitions"] = {comp["panel_ref"]: comp for comp in components}

    entry["concept_refs"] = []
    for term in entry["archdrw"]["archdrw:extensionTerms"] + entry["archdrw"]["archdrw:globalConceptLabels"]:
        qname = term_map.get(normalize_token(term))
        if qname and qname not in entry["concept_refs"]:
            entry["concept_refs"].append(qname)

    return entry


def record_to_ttl_block(rec: dict[str, Any], review: dict[str, Any], term_map: dict[str, str]) -> str:
    iid = str(rec.get("instance_id") or "").strip()
    if not iid:
        return ""

    dc_map = dict(rec.get("dublin_core") or {})
    arch_map = dict(rec.get("archdrw") or {})

    subject = draw_iri(iid)
    lines: list[str] = []
    panel_blocks: list[str] = []

    def triple(pred: str, obj: str) -> None:
        lines.append(f"    {pred} {obj} ;")

    triple("a", "archdrw:ArchivalDrawing")
    triple("dc:identifier", lit(iid))
    triple("archdrw:instanceId", lit(iid))

    title = str(dc_map.get("dc:title") or rec.get("title") or "").strip()
    if title:
        triple("dc:title", lit(title))

    creator = str(dc_map.get("dc:creator") or "").strip()
    if creator:
        triple("dc:creator", lit(creator))

    dtype = str(dc_map.get("dc:type") or rec.get("type") or "drawing").strip() or "drawing"
    triple("dc:type", lit(dtype))

    date_value = dc_map.get("dc:date") or rec.get("year")
    if date_value not in (None, ""):
        triple("dc:date", lit(str(date_value)))

    dc_format = str(dc_map.get("dc:format") or "image/jpeg").strip() or "image/jpeg"
    triple("dc:format", lit(dc_format))

    source = str(rec.get("url") or dc_map.get("dc:source") or "").strip()
    if source:
        triple("dc:source", f"<{source}>")

    relation = str(dc_map.get("dc:relation") or rec.get("project_key") or "").strip()
    if relation:
        triple("dc:relation", lit(relation))
        triple("archdrw:competition", lit(relation))

    coverage = str(dc_map.get("dc:coverage") or "").strip()
    if coverage:
        triple("dc:coverage", lit(coverage))

    rights = str(dc_map.get("dc:rights") or "").strip()
    if rights:
        triple("dc:rights", lit(rights))

    page = rec.get("page")
    if page is not None:
        triple("archdrw:boardPage", lit_int(page))

    board_layout = review.get("board_layout")
    if isinstance(board_layout, dict):
        layout_type = str(board_layout.get("layout_type") or "").strip()
        board_orientation = str(board_layout.get("board_orientation") or "").strip()
        if layout_type:
            triple("archdrw:layoutType", lit(layout_type))
        if board_orientation:
            triple("archdrw:boardOrientation", lit(board_orientation))

    # Use review output where possible.
    subjects = review.get("dc_subject")
    if not isinstance(subjects, list) or not subjects:
        subjects = dc_map.get("dc:subject") or []
    if isinstance(subjects, list):
        for item in subjects:
            value = str(item).strip().lower()
            if value:
                triple("dc:subject", lit(value))

    desc = str(review.get("dc_description") or dc_map.get("dc:description") or "").strip()
    if desc:
        triple("dc:description", lit(desc))

    # Keep known archdrw datatype fields from enriched metadata.
    for field in ("medium", "drawingStyle", "siteContext", "projection"):
        value = str(arch_map.get(field) or "").strip()
        if value:
            triple(f"archdrw:{field}", lit(value))

    for color in (arch_map.get("colorPalette") or []):
        value = str(color).strip()
        if value:
            triple("archdrw:colorPalette", lit(value))

    # Map review extension terms to schema terms from both archgen and archdrw.
    all_terms: list[str] = []
    ext_terms = review.get("extension_terms")
    if isinstance(ext_terms, list):
        all_terms.extend([str(x) for x in ext_terms])

    global_concepts = review.get("global_concepts")
    if isinstance(global_concepts, list):
        all_terms.extend([str(x) for x in global_concepts])

    seen_terms: set[str] = set()
    for term in all_terms:
        qname = term_map.get(normalize_token(str(term)))
        if not qname or qname in seen_terms:
            continue
        seen_terms.add(qname)
        if qname.startswith("archdrw:") and any(
            tag in qname.lower() for tag in ("plan", "section", "elevation", "perspective", "diagram", "map")
        ):
            triple("archdrw:drawingType", qname)
        else:
            triple("archdrw:hasVisualElement", qname)

    # Emit sub-drawing components so panels are queryable entities.
    components = review.get("components")
    if isinstance(components, list) and components:
        triple("a", "archdrw:PresentationBoard")
        for idx, comp in enumerate(components, start=1):
            if not isinstance(comp, dict):
                continue
            panel = normalize_component(comp, idx, iid, term_map, rec.get("year"))
            panel_ref = f"<{panel['panel_ref']}>"
            triple("archdrw:containsPanel", panel_ref)

            panel_lines: list[str] = [
                f"{panel_ref}",
                "    a archdrw:BoardComponent ;",
                f"    a {panel['panel_type_qname']} ;",
            ]

            panel_title = panel["panel_title"]
            if panel_title:
                panel_lines.append(f"    archdrw:panelTitle {lit(panel_title)} ;")

            if date_value not in (None, ""):
                panel_lines.append(f"    dc:date {lit(str(date_value))} ;")

            panel_order = panel["panel_order"]
            if panel_order not in (None, ""):
                panel_lines.append(f"    archdrw:panelOrder {lit_int(panel_order)} ;")

            bbox = panel["bbox"]
            if bbox:
                panel_lines.append(f"    archdrw:spatialScale {lit(f'region:{bbox}')} ;")

            ocr_text = panel["ocr_text"]
            if ocr_text:
                panel_lines.append(f"    archdrw:hasTextTranscript {lit(ocr_text)} ;")

            local_description = panel["local_description"]
            if local_description:
                panel_lines.append(f"    rdfs:comment {lit(local_description)} ;")

            for qname in panel["extracted_terms"]:
                if qname.startswith("archdrw:"):
                    panel_lines.append(f"    archdrw:usesGraphicEncoding {qname} ;")

            panel_lines[-1] = panel_lines[-1].rstrip(" ;") + " ."
            panel_blocks.append("\n".join(panel_lines) + "\n")

    if not lines:
        return ""

    lines[-1] = lines[-1].rstrip(" ;") + " ."
    main_block = f"{subject}\n" + "\n".join(lines) + "\n"
    return main_block + ("\n" + "\n".join(panel_blocks) if panel_blocks else "")


def main() -> None:
    args = parse_args()
    load_local_env()

    input_path = Path(args.input)
    schema_path = Path(args.schema)
    output_path = Path(args.output)
    json_output_path = Path(args.json_output) if args.json_output else None
    boards_root = Path(args.boards_root)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not schema_path.exists():
        raise SystemExit(f"Schema file not found: {schema_path}")
    if not boards_root.exists():
        raise SystemExit(f"Boards root not found: {boards_root}")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not args.no_openai and not api_key:
        raise SystemExit("OPENAI_API_KEY is required (or run with --no-openai)")

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input JSON must be an array")

    records = [r for r in records if isinstance(r, dict) and r.get("instance_id")]
    board_index, board_counts = load_board_index(boards_root, args.min_year, args.max_year)
    eligible_records, unmatched_records = attach_local_board_metadata(records, board_index)
    sampled = sample_by_year(eligible_records, max(1, args.max_per_year), args.seed)
    if args.max_instances and args.max_instances > 0:
        sampled = sampled[: args.max_instances]

    schema_text = schema_path.read_text(encoding="utf-8")
    term_map = parse_schema_terms(schema_text)

    print(f"Loaded records: {len(records)}")
    print(f"Eligible local boards: {sum(board_counts.values())} across years {args.min_year}-{args.max_year}")
    print(f"Matched records: {len(eligible_records)}")
    print(f"Unmatched records dropped: {unmatched_records}")
    print(f"Sampled records: {len(sampled)} (max {args.max_per_year} per year)")
    print(f"Available schema terms: {len(term_map)}")

    blocks: list[str] = []
    json_entries: list[dict[str, Any]] = []
    failures = 0

    for idx, rec in enumerate(sampled, start=1):
        review: dict[str, Any] = {}
        if not args.no_openai:
            try:
                review = call_openai_review(api_key, args.model, rec)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"WARN: OpenAI review failed for {rec.get('instance_id')}: {exc}")
            if not review.get("components"):
                print(f"WARN: no components extracted for {rec.get('instance_id')}")
        block = record_to_ttl_block(rec, review, term_map)
        if block:
            blocks.append(block)
        json_entries.append(build_drawing_json_entry(rec, review, term_map, args.panel_data_mode))
        if idx % max(1, args.log_every) == 0:
            print(f"Processed {idx}/{len(sampled)}")

    header = [
        "# extensions_sampled.ttl",
        f"# Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        f"# Input records: {len(records)}",
        f"# Sampled records: {len(sampled)}",
        "",
        "@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl:     <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .",
        f"@prefix dc:      <{DC_NS}> .",
        f"@prefix archdrw: <{ARCHDRW_NS}> .",
        f"@prefix archgen: <{ARCHGEN_NS}> .",
        "",
        "<https://archivingresearch.org/data/extensions_sampled>",
        "    a owl:Ontology ;",
        "    rdfs:label \"Sampled extension instances\"@en ;",
        f"    owl:imports <{ARCHDRW_NS.rstrip('#')}> ;",
        "    owl:imports <http://purl.org/dc/terms/> ;",
        "    owl:imports <https://archivingresearch.org/schema/drawing/extensions/generated> .",
        "",
    ]

    ttl_text = "\n".join(header + blocks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ttl_text, encoding="utf-8")

    if json_output_path:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(json_entries, indent=2), encoding="utf-8")

    print(f"Written -> {output_path}")
    if json_output_path:
        print(f"Structured JSON -> {json_output_path} ({args.panel_data_mode})")
    print(f"Blocks written: {len(blocks)}")
    if failures:
        print(f"OpenAI failures: {failures}")


if __name__ == "__main__":
    main()
