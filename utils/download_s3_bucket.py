#!/usr/bin/env python3
"""Download every object from an S3 bucket into a local directory.

Examples:
  python utils/download_s3_bucket.py --bucket my-bucket --output-dir ./downloads
  python utils/download_s3_bucket.py --bucket my-bucket --prefix docs/ --skip-existing
  python utils/download_s3_bucket.py --bucket my-bucket --profile my-aws-profile --region us-east-1
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.config import Config
from botocore import UNSIGNED
from botocore.exceptions import ClientError, NoCredentialsError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all objects from an S3 bucket (or prefix) into a local folder.",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name.")
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Local directory where files will be saved. Default: ./downloads",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional key prefix to limit downloads (e.g. 'documents/').",
    )
    parser.add_argument("--region", help="AWS region, e.g. us-east-1.")
    parser.add_argument("--profile", help="AWS profile name to use.")
    parser.add_argument(
        "--endpoint-url",
        help="Optional custom endpoint URL (useful for S3-compatible storage).",
    )
    parser.add_argument(
        "--no-sign-request",
        action="store_true",
        help="Use unsigned requests for public buckets (no AWS credentials required).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without downloading them.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def build_s3_client(
    profile: Optional[str],
    region: Optional[str],
    endpoint_url: Optional[str],
    no_sign_request: bool,
):
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)

    # Enable adaptive retry mode for transient network/API issues.
    config_kwargs = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
    if no_sign_request:
        config_kwargs["signature_version"] = UNSIGNED

    config = Config(**config_kwargs)

    return session.client("s3", region_name=region, endpoint_url=endpoint_url, config=config)


def iter_object_keys(s3_client, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip pseudo-directory keys.
            if key.endswith("/"):
                continue
            yield key


def safe_destination(base_dir: Path, key: str) -> Path:
    # Ensure all keys resolve under the selected output directory.
    destination = (base_dir / key).resolve()
    base_resolved = base_dir.resolve()
    if not str(destination).startswith(str(base_resolved)):
        raise ValueError(f"Unsafe S3 key path: {key}")
    return destination


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    s3 = build_s3_client(args.profile, args.region, args.endpoint_url, args.no_sign_request)

    total = 0
    downloaded = 0
    skipped = 0

    logging.info("Listing objects from s3://%s/%s", args.bucket, args.prefix)

    try:
        for key in iter_object_keys(s3, args.bucket, args.prefix):
            total += 1
            destination = safe_destination(output_root, key)

            if args.skip_existing and destination.exists():
                skipped += 1
                logging.debug("Skipping existing: %s", destination)
                continue

            if args.dry_run:
                logging.info("[dry-run] s3://%s/%s -> %s", args.bucket, key, destination)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(args.bucket, key, str(destination))
            downloaded += 1

            if downloaded % 100 == 0:
                logging.info("Downloaded %d files...", downloaded)
    except NoCredentialsError:
        logging.error(
            "AWS credentials not found. Configure credentials (aws configure / AWS_PROFILE) or rerun with --no-sign-request for public buckets."
        )
        return 2
    except ClientError as exc:
        logging.error("AWS client error: %s", exc)
        logging.error(
            "If this bucket is public, retry with --no-sign-request. If private, confirm your credentials and permissions."
        )
        return 3

    logging.info("Done. Objects seen: %d | Downloaded: %d | Skipped: %d", total, downloaded, skipped)

    if total == 0:
        logging.warning("No objects found. Check bucket/prefix and AWS permissions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
