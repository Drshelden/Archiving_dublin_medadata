import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .config import API_CACHE_DIR, CACHE_DIR, ROOT
from .feature_extractor import extract_board_title, extract_ocr_text, extract_visual_features, fetch_image

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

CACHE_BOARD = API_CACHE_DIR / "board_title"
CACHE_META = API_CACHE_DIR / "image_metadata"
CACHE_EXPLAIN = API_CACHE_DIR / "explain_match"
for folder in [CACHE_BOARD, CACHE_META, CACHE_EXPLAIN]:
    folder.mkdir(parents=True, exist_ok=True)

NAVIGATION_CORPUS_CANDIDATES = [
    ROOT / "data" / "processed" / "navigation_corpus_1948_2007_20py.json",
    ROOT / "frontend" / "public" / "data" / "navigation_corpus_1948_2007_20py.json",
]

_NAVIGATION_CORPUS_CACHE: Optional[Dict[str, Any]] = None
_TOKEN_RE = re.compile(r"[a-z0-9]+")

PANEL_RELATION_COLORS: Dict[str, str] = {
    "panel_type_match": "#ef4444",
    "concept_overlap": "#0ea5e9",
    "term_overlap": "#22c55e",
    "text_overlap": "#eab308",
    "same_drawing_context": "#a855f7",
    "year_proximity": "#f97316",
}

PANEL_RELATION_LABELS: Dict[str, str] = {
    "panel_type_match": "Same panel type",
    "concept_overlap": "Shared concepts",
    "term_overlap": "Shared extracted terms",
    "text_overlap": "Text/topic overlap",
    "same_drawing_context": "Same drawing context",
    "year_proximity": "Nearby year",
}


class BoardTitleRequest(BaseModel):
    image_url: str
    use_openai: bool = True


class ImageMetadataRequest(BaseModel):
    image_url: str
    instance_id: Optional[str] = None
    use_openai: bool = True
    force_refresh: bool = False


class ExplainMatchRequest(BaseModel):
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    connection_type: str
    confidence_score: float = 0.0
    evidence_kinds: list[str] = Field(default_factory=list)
    source_region_metrics: Dict[str, Any] = Field(default_factory=dict)
    target_region_metrics: Dict[str, Any] = Field(default_factory=dict)
    existing_explanation: Optional[str] = None
    use_openai: bool = True


class PanelQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=24, ge=1, le=200)
    max_edges: int = Field(default=240, ge=1, le=2000)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    selectedContext: Optional[Dict[str, Any]] = None


app = FastAPI(title="Archive AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _hash_key(payload: Dict[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(packed.encode("utf-8")).hexdigest()


def _cache_get(folder: Path, key: str) -> Optional[Dict[str, Any]]:
    path = folder / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_set(folder: Path, key: str, payload: Dict[str, Any]) -> None:
    path = folder / f"{key}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _openai_chat(system_prompt: str, user_prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=body, timeout=25)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    return None


def _extract_dc_with_openai(
    image_url: str,
    instance_id: Optional[str],
    board_title: Optional[str],
    ocr_text: str,
    visual_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None

    system = (
        "You extract Dublin Core metadata from architecture board images. "
        "Return only strict JSON with exactly these keys: "
        "dc:title, dc:creator, dc:subject, dc:description, dc:date, dc:type, dc:format, "
        "dc:identifier, dc:source, dc:relation, dc:coverage, dc:rights, dc:contributor, "
        "dc:language, dc:publisher. "
        "Rules: dc:subject must be an array of short keywords; unknown fields must be empty string, "
        "except dc:subject which must be []. Keep values concise and factual."
    )

    user = {
        "instance_id": instance_id,
        "image_url": image_url,
        "board_title": board_title,
        "ocr_text_preview": (ocr_text or "")[:4500],
        "visual_summary": visual_summary,
    }

    candidate = _openai_chat(system, json.dumps(user, ensure_ascii=False))
    if not candidate:
        return None

    parsed = _parse_json_object(candidate)
    if not parsed:
        return None

    normalized: Dict[str, Any] = {
        "dc:title": str(parsed.get("dc:title") or "").strip(),
        "dc:creator": str(parsed.get("dc:creator") or "").strip(),
        "dc:description": str(parsed.get("dc:description") or "").strip(),
        "dc:date": str(parsed.get("dc:date") or "").strip(),
        "dc:type": str(parsed.get("dc:type") or "").strip(),
        "dc:format": str(parsed.get("dc:format") or "").strip(),
        "dc:identifier": str(parsed.get("dc:identifier") or "").strip(),
        "dc:source": str(parsed.get("dc:source") or "").strip(),
        "dc:relation": str(parsed.get("dc:relation") or "").strip(),
        "dc:coverage": str(parsed.get("dc:coverage") or "").strip(),
        "dc:rights": str(parsed.get("dc:rights") or "").strip(),
        "dc:contributor": str(parsed.get("dc:contributor") or "").strip(),
        "dc:language": str(parsed.get("dc:language") or "").strip(),
        "dc:publisher": str(parsed.get("dc:publisher") or "").strip(),
    }

    subject = parsed.get("dc:subject")
    if isinstance(subject, list):
        normalized["dc:subject"] = [str(s).strip() for s in subject if str(s).strip()]
    elif isinstance(subject, str) and subject.strip():
        normalized["dc:subject"] = [subject.strip()]
    else:
        normalized["dc:subject"] = []

    if not normalized["dc:identifier"] and instance_id:
        normalized["dc:identifier"] = instance_id
    if not normalized["dc:source"]:
        normalized["dc:source"] = image_url
    if not normalized["dc:title"] and board_title:
        normalized["dc:title"] = board_title

    return normalized


def _heuristic_explanation(payload: ExplainMatchRequest) -> str:
    src = payload.source_region_metrics or {}
    tgt = payload.target_region_metrics or {}
    edge = max(float(src.get("edge_density", 0.0)), float(tgt.get("edge_density", 0.0)))
    line = max(float(src.get("line_density", 0.0)), float(tgt.get("line_density", 0.0)))
    blank = max(float(src.get("blankness", 0.0)), float(tgt.get("blankness", 0.0)))
    text = bool(src.get("text_presence")) or bool(tgt.get("text_presence"))

    phrases = []
    if edge >= 0.03:
        phrases.append("edge and corner structure")
    if line >= 0.12:
        phrases.append("line density and direction")
    if text:
        phrases.append("text-like visual patterns")
    if blank >= 0.7:
        phrases.append("large blank board areas")

    if not phrases:
        phrases.append("local visual descriptors")

    return (
        f"This match is driven by {'; '.join(phrases)}. "
        f"Confidence is {payload.confidence_score:.2f}, so treat this as a visual hint rather than a semantic guarantee."
    )


def _find_navigation_corpus_path() -> Optional[Path]:
    for path in NAVIGATION_CORPUS_CANDIDATES:
        if path.exists():
            return path
    return None


def _load_navigation_corpus() -> Dict[str, Any]:
    global _NAVIGATION_CORPUS_CACHE
    if _NAVIGATION_CORPUS_CACHE is not None:
        return _NAVIGATION_CORPUS_CACHE

    path = _find_navigation_corpus_path()
    if not path:
        return {"drawings": [], "panels": [], "concepts": []}

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"drawings": [], "panels": [], "concepts": []}

    if not isinstance(parsed, dict):
        return {"drawings": [], "panels": [], "concepts": []}

    parsed.setdefault("drawings", [])
    parsed.setdefault("panels", [])
    parsed.setdefault("concepts", [])
    _NAVIGATION_CORPUS_CACHE = parsed
    return parsed


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


def _drawing_subject_set(drawing: Dict[str, Any]) -> set[str]:
    dc = drawing.get("dc") or {}
    subject = dc.get("dc:subject")
    if not isinstance(subject, list):
        return set()
    return {str(x).strip().lower() for x in subject if str(x).strip()}


def _drawing_panel_type_set(drawing: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for panel in drawing.get("panels") or []:
        qname = str(panel.get("panel_type_qname") or "").strip()
        if qname:
            out.add(qname)
    return out


def _drawing_title_tokens(drawing: Dict[str, Any]) -> set[str]:
    dc = drawing.get("dc") or {}
    title = str(dc.get("dc:title") or "").strip()
    return _tokens(title)


def _panel_text_tokens(panel: Dict[str, Any]) -> set[str]:
    parts = [
        str(panel.get("panel_title") or ""),
        str(panel.get("local_description") or ""),
        str(panel.get("ocr_text") or ""),
    ]
    return _tokens(" ".join(parts))


def _panel_feature_text(panel: Dict[str, Any]) -> str:
    arch = panel.get("archdrw") or {}
    inherited = panel.get("inherited_context") or {}
    dc = inherited.get("dc") or {}
    inherited_arch = inherited.get("archdrw") or {}
    parts = [
        str(panel.get("panel_title") or ""),
        str(panel.get("local_description") or ""),
        str(panel.get("ocr_text") or ""),
        str(panel.get("panel_type_qname") or ""),
        " ".join(str(x) for x in (panel.get("extracted_terms") or []) if str(x).strip()),
        " ".join(str(x) for x in (panel.get("concept_refs") or []) if str(x).strip()),
        str(arch.get("rdfs:comment") or ""),
        str(dc.get("dc:title") or ""),
        str(dc.get("dc:description") or ""),
        " ".join(str(x) for x in (dc.get("dc:subject") or []) if str(x).strip()),
        str(inherited_arch.get("archdrw:competition") or ""),
        str(inherited_arch.get("archdrw:layoutType") or ""),
        str(inherited_arch.get("archdrw:boardOrientation") or ""),
    ]
    return " ".join(parts).strip()


def _panel_vector(panel: Dict[str, Any]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    text_tokens = _tokens(_panel_feature_text(panel))
    for token in text_tokens:
        vec[f"tok:{token}"] = vec.get(f"tok:{token}", 0.0) + 1.0

    ptype = str(panel.get("panel_type_qname") or "").strip()
    if ptype:
        vec[f"ptype:{ptype}"] = vec.get(f"ptype:{ptype}", 0.0) + 3.0

    for term in panel.get("extracted_terms") or []:
        t = str(term).strip()
        if t:
            vec[f"term:{t}"] = vec.get(f"term:{t}", 0.0) + 1.8

    for concept in panel.get("concept_refs") or []:
        c = str(concept).strip()
        if c:
            vec[f"concept:{c}"] = vec.get(f"concept:{c}", 0.0) + 2.0

    year = panel.get("year")
    try:
        y = int(year)
        decade = (y // 10) * 10
        vec[f"year:{y}"] = vec.get(f"year:{y}", 0.0) + 0.4
        vec[f"decade:{decade}"] = vec.get(f"decade:{decade}", 0.0) + 0.8
    except Exception:
        pass

    return vec


def _query_vector(query: str) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for token in _tokens(query):
        vec[f"tok:{token}"] = vec.get(f"tok:{token}", 0.0) + 1.0
    return vec


def _cosine_sparse(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left

    dot = 0.0
    for key, value in left.items():
        dot += value * right.get(key, 0.0)

    if dot <= 0:
        return 0.0

    ln = math.sqrt(sum(v * v for v in left.values()))
    rn = math.sqrt(sum(v * v for v in right.values()))
    if ln <= 0 or rn <= 0:
        return 0.0
    return dot / (ln * rn)


def _relation_type_between_panels(left: Dict[str, Any], right: Dict[str, Any]) -> tuple[str, float]:
    lp = str(left.get("panel_type_qname") or "").strip()
    rp = str(right.get("panel_type_qname") or "").strip()
    if lp and rp and lp == rp:
        return "panel_type_match", 1.0

    lconcept = {str(x).strip() for x in (left.get("concept_refs") or []) if str(x).strip()}
    rconcept = {str(x).strip() for x in (right.get("concept_refs") or []) if str(x).strip()}
    if lconcept and rconcept:
        overlap = _jaccard(lconcept, rconcept)
        if overlap > 0:
            return "concept_overlap", overlap

    lterms = {str(x).strip() for x in (left.get("extracted_terms") or []) if str(x).strip()}
    rterms = {str(x).strip() for x in (right.get("extracted_terms") or []) if str(x).strip()}
    if lterms and rterms:
        overlap = _jaccard(lterms, rterms)
        if overlap > 0:
            return "term_overlap", overlap

    ltext = _panel_text_tokens(left)
    rtext = _panel_text_tokens(right)
    text_overlap = _jaccard(ltext, rtext)
    if text_overlap > 0.06:
        return "text_overlap", text_overlap

    lctx = left.get("inherited_context") or {}
    rctx = right.get("inherited_context") or {}
    if str(lctx.get("drawing_ref") or "") and str(lctx.get("drawing_ref") or "") == str(rctx.get("drawing_ref") or ""):
        return "same_drawing_context", 0.8

    ly = left.get("year")
    ry = right.get("year")
    try:
        diff = abs(int(ly) - int(ry))
        if diff <= 5:
            return "year_proximity", max(0.05, 1.0 - (diff / 5.0))
    except Exception:
        pass

    return "text_overlap", 0.01


_CHAT_KEYWORD_TERMS: Dict[str, list[str]] = {
    "water": ["water", "river", "shore", "canal", "flood", "waterfront", "basin", "hydrology"],
    "vegetation": ["vegetation", "plants", "tree", "trees", "botanical", "greenery", "landscape", "canopy"],
    "topography": ["topography", "terrain", "contour", "slope", "section", "site", "relief", "earth"],
    "building": ["building", "architecture", "facade", "elevation", "envelope", "structure", "masonry", "roof"],
    "planning": ["masterplan", "urban", "circulation", "public", "space", "street", "district", "infrastructure"],
}


def _chat_search_fallback(query: str) -> Dict[str, Any]:
    q = (query or "").strip().lower()
    tokens = _tokens(q)

    terms: list[str] = []
    for keyword, related in _CHAT_KEYWORD_TERMS.items():
        if keyword in q or keyword in tokens:
            terms.extend(related)

    # Keep user tokens in ranking terms so niche queries survive.
    terms.extend([t for t in tokens if len(t) >= 3])

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        norm = str(term).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)

    search_enabled = bool(deduped)
    return {
        "enabled": search_enabled,
        "query": query,
        "terms": deduped[:16],
    }


def _extract_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_json_object(raw)
    if parsed:
        return parsed
    return None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    corpus_path = _find_navigation_corpus_path()
    corpus = _load_navigation_corpus()
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "navigation_corpus_loaded": bool(corpus.get("drawings")),
        "navigation_corpus_path": str(corpus_path) if corpus_path else None,
    }


@app.get("/api/image-proxy")
def image_proxy(url: str = Query(..., min_length=8, max_length=4096)) -> Response:
    parsed = str(url or "").strip()
    if not parsed.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http/https image URLs are supported")

    try:
        upstream = requests.get(parsed, timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream image request failed: {exc}")

    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail="Upstream image request returned an error")

    content_type = upstream.headers.get("content-type", "application/octet-stream")
    return Response(
        content=upstream.content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/similar-drawings")
def similar_drawings(instance_id: Optional[str] = None, drawing_ref: Optional[str] = None, top_k: int = 12) -> Dict[str, Any]:
    corpus = _load_navigation_corpus()
    drawings = [d for d in corpus.get("drawings", []) if isinstance(d, dict)]
    if not drawings:
        return {"ok": False, "error": "Navigation corpus not available", "results": []}

    query: Optional[Dict[str, Any]] = None
    if instance_id:
        query = next((d for d in drawings if str(d.get("instance_id") or "") == instance_id), None)
    if query is None and drawing_ref:
        query = next((d for d in drawings if str(d.get("drawing_ref") or "") == drawing_ref), None)
    if query is None:
        return {"ok": False, "error": "Query drawing not found", "results": []}

    q_year = query.get("year")
    q_subject = _drawing_subject_set(query)
    q_concepts = {str(x).strip() for x in (query.get("concept_refs") or []) if str(x).strip()}
    q_panel_types = _drawing_panel_type_set(query)
    q_title = _drawing_title_tokens(query)

    ranked = []
    for candidate in drawings:
        if candidate is query:
            continue

        c_subject = _drawing_subject_set(candidate)
        c_concepts = {str(x).strip() for x in (candidate.get("concept_refs") or []) if str(x).strip()}
        c_panel_types = _drawing_panel_type_set(candidate)
        c_title = _drawing_title_tokens(candidate)

        subject_overlap = _jaccard(q_subject, c_subject)
        concept_overlap = _jaccard(q_concepts, c_concepts)
        panel_overlap = _jaccard(q_panel_types, c_panel_types)
        title_overlap = _jaccard(q_title, c_title)

        year_score = 0.0
        cyear = candidate.get("year")
        if q_year is not None and cyear is not None:
            try:
                year_diff = abs(int(q_year) - int(cyear))
                year_score = max(0.0, 1.0 - (year_diff / 20.0))
            except Exception:
                year_score = 0.0

        score = (
            0.35 * concept_overlap
            + 0.25 * panel_overlap
            + 0.2 * subject_overlap
            + 0.1 * title_overlap
            + 0.1 * year_score
        )

        reasons = []
        if concept_overlap > 0:
            reasons.append("concept overlap")
        if panel_overlap > 0:
            reasons.append("panel type overlap")
        if subject_overlap > 0:
            reasons.append("subject overlap")
        if year_score >= 0.7:
            reasons.append("nearby year")

        ranked.append(
            {
                "drawing_ref": candidate.get("drawing_ref"),
                "instance_id": candidate.get("instance_id"),
                "year": candidate.get("year"),
                "title": (candidate.get("dc") or {}).get("dc:title"),
                "score": round(score, 4),
                "reasons": reasons,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    limit = max(1, min(int(top_k), 100))
    return {
        "ok": True,
        "query": {
            "drawing_ref": query.get("drawing_ref"),
            "instance_id": query.get("instance_id"),
            "title": (query.get("dc") or {}).get("dc:title"),
            "year": query.get("year"),
        },
        "results": ranked[:limit],
    }


@app.get("/api/similar-panels")
def similar_panels(panel_ref: str, top_k: int = 16) -> Dict[str, Any]:
    corpus = _load_navigation_corpus()
    panels = [p for p in corpus.get("panels", []) if isinstance(p, dict)]
    if not panels:
        return {"ok": False, "error": "Navigation corpus not available", "results": []}

    query = next((p for p in panels if str(p.get("panel_ref") or "") == panel_ref), None)
    if query is None:
        return {"ok": False, "error": "Query panel not found", "results": []}

    q_type = str(query.get("panel_type_qname") or "").strip()
    q_terms = {str(x).strip() for x in (query.get("extracted_terms") or []) if str(x).strip()}
    q_concepts = {str(x).strip() for x in (query.get("concept_refs") or []) if str(x).strip()}
    q_text = _panel_text_tokens(query)
    q_year = query.get("year")

    ranked = []
    for candidate in panels:
        if candidate is query:
            continue

        c_type = str(candidate.get("panel_type_qname") or "").strip()
        c_terms = {str(x).strip() for x in (candidate.get("extracted_terms") or []) if str(x).strip()}
        c_concepts = {str(x).strip() for x in (candidate.get("concept_refs") or []) if str(x).strip()}
        c_text = _panel_text_tokens(candidate)

        type_score = 1.0 if q_type and q_type == c_type else 0.0
        term_overlap = _jaccard(q_terms, c_terms)
        concept_overlap = _jaccard(q_concepts, c_concepts)
        text_overlap = _jaccard(q_text, c_text)

        year_score = 0.0
        cyear = candidate.get("year")
        if q_year is not None and cyear is not None:
            try:
                year_diff = abs(int(q_year) - int(cyear))
                year_score = max(0.0, 1.0 - (year_diff / 20.0))
            except Exception:
                year_score = 0.0

        score = (
            0.45 * type_score
            + 0.2 * concept_overlap
            + 0.15 * term_overlap
            + 0.1 * text_overlap
            + 0.1 * year_score
        )

        inherited = candidate.get("inherited_context") or {}
        ranked.append(
            {
                "panel_ref": candidate.get("panel_ref"),
                "panel_id": candidate.get("panel_id"),
                "panel_title": candidate.get("panel_title"),
                "panel_type_qname": c_type,
                "year": candidate.get("year"),
                "drawing_ref": inherited.get("drawing_ref"),
                "instance_id": inherited.get("instance_id"),
                "score": round(score, 4),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    limit = max(1, min(int(top_k), 100))
    query_inherited = query.get("inherited_context") or {}
    return {
        "ok": True,
        "query": {
            "panel_ref": query.get("panel_ref"),
            "panel_id": query.get("panel_id"),
            "panel_title": query.get("panel_title"),
            "panel_type_qname": query.get("panel_type_qname"),
            "drawing_ref": query_inherited.get("drawing_ref"),
            "instance_id": query_inherited.get("instance_id"),
            "year": query.get("year"),
        },
        "results": ranked[:limit],
    }


@app.post("/api/panel-query-graph")
def panel_query_graph(req: PanelQueryRequest) -> Dict[str, Any]:
    corpus = _load_navigation_corpus()
    panels = [p for p in corpus.get("panels", []) if isinstance(p, dict)]
    if not panels:
        return {"ok": False, "error": "Navigation corpus not available", "graph": {"nodes": [], "edges": [], "connection_color_map": {}}}

    qvec = _query_vector(req.query)
    ranked: list[tuple[float, Dict[str, Any]]] = []
    for panel in panels:
        pvec = _panel_vector(panel)
        score = _cosine_sparse(qvec, pvec)
        if score >= req.min_score:
            ranked.append((score, panel))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[: req.top_k]

    nodes = []
    selected_panels_by_ref: Dict[str, Dict[str, Any]] = {}
    for score, panel in selected:
        pref = str(panel.get("panel_ref") or "").strip()
        if not pref:
            continue
        selected_panels_by_ref[pref] = panel
        inherited = panel.get("inherited_context") or {}
        dc = inherited.get("dc") or {}
        label = str(panel.get("panel_title") or "").strip() or str(dc.get("dc:title") or "").strip() or str(panel.get("panel_id") or pref)
        nodes.append(
            {
                "id": pref,
                "panel_ref": pref,
                "instance_id": inherited.get("instance_id"),
                "drawing_ref": inherited.get("drawing_ref"),
                "label": label,
                "cluster": str(panel.get("panel_type_qname") or "panel"),
                "score": round(float(score), 4),
                "year": panel.get("year"),
            }
        )

    edges = []
    refs = list(selected_panels_by_ref.keys())
    edge_idx = 0
    for i in range(len(refs)):
        left_ref = refs[i]
        left_panel = selected_panels_by_ref[left_ref]
        for j in range(i + 1, len(refs)):
            right_ref = refs[j]
            right_panel = selected_panels_by_ref[right_ref]
            rel_type, rel_strength = _relation_type_between_panels(left_panel, right_panel)
            if rel_strength < 0.08:
                continue
            edge_idx += 1
            edges.append(
                {
                    "id": f"panel_rel_{edge_idx}",
                    "source": left_ref,
                    "target": right_ref,
                    "weight": round(float(rel_strength), 4),
                    "connection_types": [rel_type],
                    "explanation": PANEL_RELATION_LABELS.get(rel_type, rel_type),
                    "color": PANEL_RELATION_COLORS.get(rel_type, "#9ca3af"),
                }
            )

    edges.sort(key=lambda item: item.get("weight", 0.0), reverse=True)
    edges = edges[: req.max_edges]

    return {
        "ok": True,
        "query": req.query,
        "top_k": req.top_k,
        "matched": len(nodes),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "connection_color_map": PANEL_RELATION_COLORS,
            "connection_labels": PANEL_RELATION_LABELS,
        },
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    cleaned_messages = [m for m in req.messages if m and isinstance(m.content, str) and m.content.strip()]
    if not cleaned_messages:
        return {
            "answer": "Please send a message so I can search the archive.",
            "search": {"enabled": False, "query": "", "terms": []},
        }

    latest_user = ""
    for m in reversed(cleaned_messages):
        if str(m.role).strip().lower() == "user":
            latest_user = m.content.strip()
            break
    if not latest_user:
        latest_user = cleaned_messages[-1].content.strip()

    fallback_search = _chat_search_fallback(latest_user)

    if OPENAI_API_KEY:
        selected = req.selectedContext or {}
        system_prompt = (
            "You are an assistant for an architecture drawing archive explorer. "
            "Return strict JSON only with shape: "
            '{"answer":"string","search":{"enabled":boolean,"query":"string","terms":["string"]}}.'
        )
        user_payload = {
            "selected_context": {
                "title": selected.get("title"),
                "instance_id": selected.get("instanceId"),
                "secondary_line": selected.get("secondaryLine"),
            },
            "latest_user_message": latest_user,
            "hint_search": fallback_search,
        }
        ai_raw = _openai_chat(system_prompt, json.dumps(user_payload, ensure_ascii=False))
        if ai_raw:
            ai_obj = _extract_json_from_text(ai_raw)
            if ai_obj and isinstance(ai_obj, dict):
                answer = str(ai_obj.get("answer") or "").strip() or "I searched the archive using your request."
                ai_search = ai_obj.get("search") if isinstance(ai_obj.get("search"), dict) else {}
                enabled = bool(ai_search.get("enabled"))
                query = str(ai_search.get("query") or latest_user).strip()
                raw_terms = ai_search.get("terms") if isinstance(ai_search.get("terms"), list) else []
                terms = [str(t).strip().lower() for t in raw_terms if str(t).strip()]
                if not terms:
                    terms = fallback_search.get("terms", [])
                return {
                    "answer": answer,
                    "search": {
                        "enabled": enabled or bool(terms),
                        "query": query,
                        "terms": terms[:16],
                    },
                }

    return {
        "answer": "I searched the archive with your query and updated the graph with relevant panels.",
        "search": fallback_search,
    }


@app.post("/api/extract-board-title")
def extract_board_title_route(req: BoardTitleRequest) -> Dict[str, Any]:
    key = _hash_key(req.model_dump())
    cached = _cache_get(CACHE_BOARD, key)
    if cached:
        return cached

    cached_path = fetch_image(CACHE_DIR, req.image_url)
    if not cached_path:
        payload = {"ok": False, "error": "Could not fetch image"}
        _cache_set(CACHE_BOARD, key, payload)
        return payload

    result = extract_board_title(cached_path)
    title = result.get("board_title")
    refined_title = title
    openai_used = False

    if req.use_openai and title:
        system = "You clean OCR board titles. Return only one cleaned title line."
        user = f"OCR title: {title}\nClean OCR mistakes but keep original wording."
        candidate = _openai_chat(system, user)
        if candidate:
            refined_title = candidate.splitlines()[0].strip()
            openai_used = True

    payload = {
        "ok": True,
        "image_url": req.image_url,
        "board_title": refined_title,
        "raw_board_title": title,
        "board_title_confidence": result.get("board_title_confidence", 0.0),
        "source": "openai" if openai_used else result.get("board_title_source", "heuristic_ocr"),
        "openai_used": openai_used,
    }
    _cache_set(CACHE_BOARD, key, payload)
    return payload


@app.post("/api/extract-image-metadata")
def extract_image_metadata_route(req: ImageMetadataRequest) -> Dict[str, Any]:
    key = _hash_key(req.model_dump())
    if not req.force_refresh:
        cached = _cache_get(CACHE_META, key)
        if cached:
            return cached

    cached_path = fetch_image(CACHE_DIR, req.image_url)
    if not cached_path:
        payload = {"ok": False, "error": "Could not fetch image"}
        _cache_set(CACHE_META, key, payload)
        return payload

    visual = extract_visual_features(cached_path)
    ocr_text = extract_ocr_text(cached_path)
    title = extract_board_title(cached_path)

    payload = {
        "ok": True,
        "instance_id": req.instance_id,
        "image_url": req.image_url,
        "board_title": title.get("board_title"),
        "board_title_confidence": title.get("board_title_confidence", 0.0),
        "ocr_text": ocr_text,
        "visual_summary": {
            "edge_density": visual.get("edge_density", 0.0),
            "keypoint_count": len(visual.get("orb_keypoints", [])),
            "width": visual.get("width"),
            "height": visual.get("height"),
        },
        "dublin_core": {
            "dc:title": title.get("board_title") or "",
            "dc:creator": "",
            "dc:subject": [],
            "dc:description": "",
            "dc:date": "",
            "dc:type": "drawing",
            "dc:format": "image/jpeg",
            "dc:identifier": req.instance_id or "",
            "dc:source": req.image_url,
            "dc:relation": "",
            "dc:coverage": "",
            "dc:rights": "",
            "dc:contributor": "",
            "dc:language": "",
            "dc:publisher": "",
        },
        "openai_used": False,
    }

    if req.use_openai and OPENAI_API_KEY:
        system = "You summarize architecture board metadata from OCR and simple metrics in one sentence."
        user = json.dumps(
            {
                "board_title": payload["board_title"],
                "ocr_text_preview": ocr_text[:400],
                "visual_summary": payload["visual_summary"],
            }
        )
        summary = _openai_chat(system, user)
        if summary:
            payload["llm_summary"] = summary

        extracted_dc = _extract_dc_with_openai(
            image_url=req.image_url,
            instance_id=req.instance_id,
            board_title=payload["board_title"],
            ocr_text=ocr_text,
            visual_summary=payload["visual_summary"],
        )
        if extracted_dc:
            payload["dublin_core"] = extracted_dc
            payload["openai_used"] = True

    _cache_set(CACHE_META, key, payload)
    return payload


@app.post("/api/explain-match")
def explain_match_route(req: ExplainMatchRequest) -> Dict[str, Any]:
    key = _hash_key(req.model_dump())
    cached = _cache_get(CACHE_EXPLAIN, key)
    if cached:
        return cached

    explanation = req.existing_explanation or _heuristic_explanation(req)
    openai_used = False

    if req.use_openai and OPENAI_API_KEY:
        system = (
            "You explain visual match evidence in plain language for architects. "
            "Use 2 concise sentences and avoid generic wording."
        )
        user = json.dumps(req.model_dump())
        candidate = _openai_chat(system, user)
        if candidate:
            explanation = candidate
            openai_used = True

    payload = {
        "ok": True,
        "source_instance_id": req.source_instance_id,
        "target_instance_id": req.target_instance_id,
        "explanation": explanation,
        "evidence_kinds": req.evidence_kinds,
        "openai_used": openai_used,
    }
    _cache_set(CACHE_EXPLAIN, key, payload)
    return payload
