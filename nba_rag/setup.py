"""Download official sources and Hugging Face models; build the persistent index."""
import argparse
from .config import DATA
from .ingest import ingest
from .engine import NBAEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download all official NBA articles")
    args = parser.parse_args()
    if args.refresh or not (DATA / "documents.json").exists():
        ingest()
    engine = NBAEngine()
    engine.load()
    print(f"Ready: {len(engine.documents)} official documents, {len(engine.chunks)} indexed passages.", flush=True)


if __name__ == "__main__":
    main()
