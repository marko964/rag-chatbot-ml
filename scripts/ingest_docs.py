#!/usr/bin/env python3
"""
Ingest PDF and TXT documents into ChromaDB.

Usage:
    python scripts/ingest_docs.py --dir data/documents
    python scripts/ingest_docs.py --file path/to/file.pdf
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingest import ingest_file, ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Single PDF or TXT file to ingest")
    group.add_argument("--dir", type=Path, help="Directory of documents to ingest")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        count = ingest_file(args.file)
        print(f"Ingested {count} chunks from '{args.file.name}'")
    else:
        if not args.dir.exists():
            print(f"Error: directory not found: {args.dir}")
            sys.exit(1)
        results = ingest_directory(args.dir)
        if not results:
            print("No supported files found (.pdf, .txt, .md)")
        else:
            total = 0
            for filename, count in results.items():
                print(f"  {filename}: {count} chunks")
                total += count
            print(f"\nTotal: {len(results)} files, {total} chunks")


if __name__ == "__main__":
    main()
