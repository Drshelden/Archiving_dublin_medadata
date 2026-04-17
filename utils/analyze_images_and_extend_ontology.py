#!/usr/bin/env python3
"""Analyze images from AWS S3 and propose ontology extensions with OpenAI Vision.

Usage:
    python utils/analyze_images_and_extend_ontology.py --no-sign-request --schema schemas/archival_drawing_revised.ttl
    python utils/analyze_images_and_extend_ontology.py --prefix some/subfolder/ --max-images 50 --no-sign-request --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import boto3
    import requests
    from botocore.config import Config
    from botocore import UNSIGNED
    from botocore.exceptions import NoCredentialsError
except ImportError:
    print("Required dependencies missing. Install with:")
    print("  pip install boto3 requests")
    sys.exit(1)


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_BUCKET = "archivingresearch"
DEFAULT_MODEL = "gpt-4o-mini"


def load_local_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for env_path in (repo_root / ".env", repo_root / "backend" / ".env"):
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
    parser = argparse.ArgumentParser(
        description="Analyze archival images and propose ontology extensions.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket containing images to analyze. Default: {DEFAULT_BUCKET}",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional S3 key prefix to limit images. Keys in this repo appear to be stored at bucket root.",
    )
    parser.add_argument(
        "--region",
        help="AWS region, e.g. us-east-1.",
    )
    parser.add_argument(
        "--profile",
        help="AWS profile name to use.",
    )
    parser.add_argument(
        "--endpoint-url",
        help="Optional custom endpoint URL (for S3-compatible storage).",
    )
    parser.add_argument(
        "--no-sign-request",
        action="store_true",
        help="Use unsigned requests for public buckets.",
    )
    parser.add_argument(
        "--schema",
        default="schemas/archival_drawing_revised.ttl",
        help="Path to ontology TTL file to extend.",
    )
    parser.add_argument(
        "--output",
        default="ontology_extensions.ttl",
        help="Output file for proposed ontology extensions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use for image analysis. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Max images to analyze (default: 10 for cost control).",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample images from all matching S3 keys.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --random-sample is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report without modifying schema.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def build_s3_client(
    profile: Optional[str],
    region: Optional[str],
    endpoint_url: Optional[str],
    no_sign_request: bool,
):
    session_kwargs: dict[str, str] = {}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)

    config_kwargs: dict[str, Any] = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
    if no_sign_request:
        config_kwargs["signature_version"] = UNSIGNED

    config = Config(**config_kwargs)
    return session.client("s3", region_name=region, endpoint_url=endpoint_url, config=config)


def iter_image_keys(s3_client, bucket: str, prefix: str) -> Iterator[str]:
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if key.lower().endswith(image_extensions):
                yield key


def download_image_bytes(s3_client, bucket: str, key: str) -> bytes:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def normalize_prefix(prefix: str) -> str:
    return prefix.lstrip("/")


def load_existing_ontology(schema_path: Path) -> dict[str, Any]:
    """Parse TTL and extract existing classes/properties."""
    content = schema_path.read_text(encoding="utf-8")
    
    result = {
        "classes": set(),
        "properties": set(),
        "raw_content": content,
    }
    
    # Simple extraction of archdrw: classes and properties
    import re
    classes = re.findall(r"archdrw:(\w+)\s+a\s+owl:Class", content)
    properties = re.findall(r"archdrw:(\w+)\s+a\s+owl:(ObjectProperty|DatatypeProperty)", content)
    
    result["classes"] = set(classes)
    result["properties"] = set(p[0] for p in properties)
    
    return result


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            parsed = json.loads(text[json_start:json_end])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"raw_response": raw}


def analyze_image_with_openai(
    image_key: str,
    image_data: bytes,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Send image bytes from S3 to OpenAI for content analysis."""

    import base64
    b64_image = base64.standard_b64encode(image_data).decode("utf-8")
    
    ext = Path(image_key).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    prompt = """Analyze this archival/design image and return JSON only with this exact object shape:

{
  "primary_content_type": "category (e.g., architectural drawing, urban map, diagram, mixed-media)",
  "visual_elements": ["list of identified visual elements (buildings, trees, annotations, etc)"],
  "document_structure": "description of layout (e.g., single panel, multi-panel, with insets)",
  "color_scheme": ["dominant colors"],
  "annotations_present": true/false,
  "scale_indicators": ["scale bars, north arrows, dimensions, etc"],
  "text_metadata": ["visible titles, legends, labels"],
  "drawing_styles": ["technical, freehand, diagrammatic, photorealistic, etc"],
  "domain_concepts": ["conceptual elements: connectedness, flows, hierarchies, etc"],
  "ontology_gaps": ["metadata not easily captured by standard archival vocabulary"]
}

Focus on metadata that should be captured semantically, not pixel-level details."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You analyze archival design images and return strict JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}",
                        },
                    },
                ],
            },
        ],
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=body, timeout=90)
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        return parse_json_object(text)
    except requests.HTTPError:
        logging.error("OpenAI analysis failed for %s: HTTP %s %s", image_key, response.status_code, response.text[:300])
        return {"error": f"HTTP {response.status_code}", "response": response.text[:1000]}
    except Exception as exc:
        logging.error(f"OpenAI analysis failed for {image_key}: {exc}")
        return {"error": str(exc)}


def propose_ontology_extensions(
    analyses: list[dict[str, Any]],
    existing_ontology: dict[str, Any],
) -> list[str]:
    """Propose new classes and properties based on analysis."""
    proposed = []
    
    for analysis in analyses:
        if "error" in analysis:
            continue
        
        # Propose classes from domain_concepts
        for concept in analysis.get("domain_concepts", []):
            if concept and not any(c.lower() == concept.lower() for c in existing_ontology["classes"]):
                class_name = "".join(w.capitalize() for w in concept.split())
                proposed.append(f"class {class_name} ({concept})")
        
        # Propose properties from ontology_gaps
        for gap in analysis.get("ontology_gaps", []):
            if gap:
                prop_name = "".join(w.capitalize() for w in gap.split())
                proposed.append(f"property has{prop_name} ({gap})")
    
    return list(set(proposed))


def generate_ttl_extensions(
    proposed: list[str],
    image_count: int,
) -> str:
    """Generate TTL snippet for proposed extensions."""
    ttl = f"""# ═══════════════════════════════════════════════════════════════════════════
# PROPOSED ONTOLOGY EXTENSIONS
# Generated from analysis of {image_count} archival images
# ═══════════════════════════════════════════════════════════════════════════

# New Classes
"""
    
    for item in proposed:
        if item.startswith("class "):
            name, desc = item.replace("class ", "").split(" (")
            desc = desc.rstrip(")")
            ttl += f"""
archdrw:{name}
    a owl:Class ;
    rdfs:label "{name}"@en ;
    rdfs:comment "{desc}"@en ;
    rdfs:subClassOf archdrw:VisualElement ."""
    
    ttl += "\n\n# New Properties\n"
    
    for item in proposed:
        if item.startswith("property "):
            name, desc = item.replace("property ", "").split(" (")
            desc = desc.rstrip(")")
            ttl += f"""
archdrw:{name}
    a owl:ObjectProperty ;
    rdfs:label "{name}"@en ;
    rdfs:comment "{desc}"@en ;
    rdfs:domain archdrw:ArchivalDrawing ;
    rdfs:range archdrw:VisualElement ."""
    
    return ttl


def main() -> int:
    load_local_env()
    args = parse_args()
    args.prefix = normalize_prefix(args.prefix)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        return 1
    
    try:
        s3_client = build_s3_client(
            profile=args.profile,
            region=args.region,
            endpoint_url=args.endpoint_url,
            no_sign_request=args.no_sign_request,
        )
    except Exception as exc:
        logging.error(f"Failed to initialize S3 client: {exc}")
        return 1
    
    schema_path = Path(args.schema)
    if not schema_path.exists():
        logging.error(f"Schema not found: {schema_path}")
        return 1
    
    # Load existing ontology
    existing_ontology = load_existing_ontology(schema_path)
    logging.info(f"Loaded ontology with {len(existing_ontology['classes'])} classes and {len(existing_ontology['properties'])} properties")
    
    # Find image keys in S3
    try:
        all_images = list(iter_image_keys(s3_client, args.bucket, args.prefix))
    except NoCredentialsError:
        logging.error(
            "No AWS credentials found. If this is a public bucket, rerun with --no-sign-request."
        )
        return 1
    except Exception as exc:
        logging.error(f"Failed to list images from s3://{args.bucket}/{args.prefix}: {exc}")
        return 1
    
    if not all_images:
        logging.warning(f"No images found in s3://{args.bucket}/{args.prefix}")
        return 0

    if args.random_sample:
        sample_size = min(args.max_images, len(all_images))
        images = random.Random(args.seed).sample(all_images, sample_size)
        logging.info(
            "Randomly sampled %s images from %s total keys using seed=%s",
            len(images),
            len(all_images),
            args.seed,
        )
    else:
        images = all_images[:args.max_images]
    
    logging.info(f"Found {len(images)} images to analyze from s3://{args.bucket}/{args.prefix}")
    
    # Analyze images
    analyses = []
    for i, image_key in enumerate(images, 1):
        logging.info(f"[{i}/{len(images)}] Analyzing {image_key}...")
        try:
            image_data = download_image_bytes(s3_client, args.bucket, image_key)
        except Exception as exc:
            logging.error(f"Failed to download s3://{args.bucket}/{image_key}: {exc}")
            analyses.append({"error": str(exc), "key": image_key})
            continue

        analysis = analyze_image_with_openai(image_key, image_data, api_key, args.model)
        analysis["source_key"] = image_key
        analyses.append(analysis)
    
    # Propose extensions
    proposed = propose_ontology_extensions(analyses, existing_ontology)
    
    if not proposed:
        logging.info("No new ontology extensions proposed.")
        return 0
    
    logging.info(f"Proposed {len(proposed)} ontology extensions:")
    for p in proposed:
        logging.info(f"  - {p}")
    
    # Generate TTL
    ttl_extensions = generate_ttl_extensions(proposed, len(images))
    
    output_path = Path(args.output)
    output_path.write_text(ttl_extensions, encoding="utf-8")
    logging.info(f"Proposed extensions written to {output_path}")
    
    if not args.dry_run:
        logging.info("To apply extensions, manually review and merge into archival_drawing_revised.ttl")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
