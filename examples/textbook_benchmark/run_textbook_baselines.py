"""Compatibility entry point for textbook benchmark baselines."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DIRECT_BASELINES = {"llm", "llm_only", "rag", "rag_serper", "llm_rag"}
OMNITHINK_BASELINES = {"omnithink", "omni_think"}


def requested_baselines(argv):
    if "--baselines" not in argv:
        return []
    index = argv.index("--baselines") + 1
    baselines = []
    while index < len(argv) and not argv[index].startswith("-"):
        baselines.append(argv[index])
        index += 1
    return baselines


def main():
    baselines = requested_baselines(sys.argv[1:])
    if any(baseline in OMNITHINK_BASELINES for baseline in baselines):
        from examples.textbook_benchmark.run_omnithink_textbook_benchmark import (
            main as omnithink_main,
        )

        return omnithink_main()

    if any(baseline in DIRECT_BASELINES for baseline in baselines):
        from examples.textbook_benchmark.run_qwen_direct_baselines import (
            main as direct_main,
        )

        return direct_main()

    from examples.storm_examples.run_storm_textbook_benchmark import main as storm_main

    return storm_main()


if __name__ == "__main__":
    raise SystemExit(main())
