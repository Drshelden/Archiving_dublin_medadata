#!/usr/bin/env python3
"""List top attribute counts from metadata records.

Default behavior counts non-empty values under each record's `archdrw` object and
reports key:value frequencies in descending order.

Use --record-field to count top values for a direct record field (for example,
--record-field title).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report top key:value attribute counts from a metadata JSON file.",
    )
    parser.add_argument(
        "--input",
        default="frontend/public/data/enriched_metadata.json",
        help="Path to input JSON file (default: frontend/public/data/enriched_metadata.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )
    parser.add_argument(
        "--section",
        default="archdrw",
        help="Record section to inspect (default: archdrw)",
    )
    parser.add_argument(
        "--record-field",
        default="",
        help="Optional direct record field to count (for example: title)",
    )
    parser.add_argument(
        "--value",
        default="",
        help="Optional exact value to count across the section (e.g. warm_tones)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be an array of records")

    section_name = args.section
    record_field = args.record_field.strip()
    value_filter = args.value.strip()
    counter: collections.Counter[str] = collections.Counter()
    value_count = 0

    for record in data:
        if record_field:
            field_value = record.get(record_field)
            if isinstance(field_value, list):
                for item in field_value:
                    text = str(item).strip()
                    if not text:
                        continue
                    counter[text] += 1
                    if value_filter and text == value_filter:
                        value_count += 1
            else:
                text = str(field_value).strip()
                if text:
                    counter[text] += 1
                    if value_filter and text == value_filter:
                        value_count += 1
            continue

        section = record.get(section_name) or {}
        if not isinstance(section, dict):
            continue

        for key, value in section.items():
            if isinstance(value, list):
                for item in value:
                    text = str(item).strip()
                    if not text:
                        continue
                    counter[f"{key}:{text}"] += 1
                    if value_filter and text == value_filter:
                        value_count += 1
            else:
                text = str(value).strip()
                if not text:
                    continue
                counter[f"{key}:{text}"] += 1
                if value_filter and text == value_filter:
                    value_count += 1

    print(f"TOTAL_RECORDS\t{len(data)}")
    if record_field:
        print(f"FIELD\t{record_field}")
    else:
        print(f"SECTION\t{section_name}")

    for label, count in counter.most_common(max(0, args.top)):
        print(f"{label}\t{count}")

    if value_filter:
        print(f"VALUE_COUNT\t{value_filter}\t{value_count}")


if __name__ == "__main__":
    main()
