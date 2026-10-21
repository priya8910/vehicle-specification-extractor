from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_spec_extractor.config import DEFAULT_PERSIST_DIRECTORY
from vehicle_spec_extractor.extraction import (
    extract_vehicle_specs,
    export_csv,
    export_json,
)
from vehicle_spec_extractor.pipeline import (
    ingest_manual,
    retrieve_manual,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vehicle-spec-extractor",
        description=(
            "Index the supplied workshop manual or extract cited specifications."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index a workshop PDF.")
    ingest_parser.add_argument("pdf", type=Path)
    ingest_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
    )

    ask_parser = subparsers.add_parser("ask", help="Retrieve and extract from a PDF.")
    ask_parser.add_argument("pdf", type=Path)
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--format", choices=("json", "csv"), default="json")
    ask_parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
    )
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            outcome = ingest_manual(
                args.pdf,
                persist_directory=args.persist_dir,
            )
            print(json.dumps(outcome, indent=2))
            return

        evidence = retrieve_manual(
            args.pdf,
            args.question,
            top_k=args.top_k,
            persist_directory=args.persist_dir,
        )
        result = extract_vehicle_specs(args.question, evidence)
        if args.format == "csv":
            print(export_csv(result), end="")
        else:
            print(export_json(result))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")


if __name__ == "__main__":
    main()
