from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_ci_cd.config import DOCS_DIR, EVAL_DIR, INDEX_DIR
from rag_ci_cd.service.app import build_index


def cmd_index():
    """Build or rebuild the search index from docs/."""
    store = build_index()
    print(f"\nDone. {store.size} chunks indexed.")


def cmd_eval():
    """Run evaluation on the gold set and print results."""
    from rag_ci_cd.evaluation.harness import GOLD_SET, print_evaluation, run_evaluation
    from rag_ci_cd.indexing.store import IndexStore

    store = IndexStore(INDEX_DIR)
    if not store.load():
        print("No index found. Run `python -m rag_ci_cd.cli index` first.")
        return

    results = run_evaluation(store, GOLD_SET)
    print_evaluation(results)

    save_path = EVAL_DIR / "eval_results.json"
    save_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {save_path}")


def cmd_serve():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("rag_ci_cd.service.app:app", host="0.0.0.0", port=6565, reload=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m rag_ci_cd.cli [index|eval|serve]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "index":
        cmd_index()
    elif command == "eval":
        cmd_eval()
    elif command == "serve":
        cmd_serve()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m rag_ci_cd.cli [index|eval|serve]")
        sys.exit(1)
