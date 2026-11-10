"""Command-line entry point for validation, evaluation, and regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.dataset import load_jsonl, validate_repository_data
from src.evaluation.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parents[2]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Golden retrieval evaluation",
        "",
        f"Dataset: `{report['dataset_version']}`  ",
        f"Golden questions: {report['golden_questions']}  ",
        f"Embedding: `{report['embedding']['model']}` ({report['embedding']['dimensions']} dimensions)",
        "",
        "| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | Evidence coverage@5 | p95 latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["configurations"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['recall_at_1']:.1%} | {metrics['recall_at_5']:.1%} | "
            f"{metrics['recall_at_10']:.1%} | {metrics['reciprocal_rank']:.3f} | "
            f"{metrics['ndcg_at_5']:.3f} | {metrics['evidence_coverage_at_5']:.1%} | "
            f"{metrics['latency_p95_ms']:.3f} |"
        )
    lines.extend(["", "## Recall@5 by question tag", ""])
    tags = sorted(
        {tag for result in report["configurations"].values() for tag in result["breakdown"]}
    )
    lines.append("| Configuration | " + " | ".join(tags) + " |")
    lines.append("|---|" + "---:|" * len(tags))
    for name, result in report["configurations"].items():
        values = [result["breakdown"].get(tag, {}).get("recall_at_5", 0.0) for tag in tags]
        lines.append(f"| {name} | " + " | ".join(f"{value:.1%}" for value in values) + " |")
    lines.extend(
        [
            "",
            "> These are measured offline results for the committed synthetic dataset. "
            "They are not production claims. Re-run the command below after retrieval or dataset changes.",
            "",
            "```bash",
            "python -m src.evaluation.cli --compare-to reports/evaluation/baseline.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _compare(report: dict[str, Any], baseline_path: Path, max_drop: float) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = []
    for name, result in report["configurations"].items():
        current = result["metrics"]["recall_at_5"]
        expected = baseline["configurations"][name]["metrics"]["recall_at_5"]
        if expected - current > max_drop:
            failures.append(f"{name}: Recall@5 {current:.3f} is below baseline {expected:.3f}")
    if failures:
        raise SystemExit("Evaluation regression:\n" + "\n".join(failures))


def _record_manifest_results(report: dict[str, Any]) -> None:
    mappings = {
        "tickets-v1.0": ["A_recursive_vector"],
        "tickets-v2.0-ticket-aware-chunking": [
            "B_ticket_aware_hybrid",
            "C_ticket_aware_hybrid_reranker",
        ],
    }
    for version, configuration_names in mappings.items():
        path = ROOT / "datasets" / version / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["evaluation_results"] = {
            "status": "measured",
            "report": "reports/evaluation/baseline.json",
            "configurations": {
                name: report["configurations"][name]["metrics"] for name in configuration_names
            },
        }
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "evaluation" / "latest.json"
    )
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--record-manifests", action="store_true")
    args = parser.parse_args()

    validation = validate_repository_data(ROOT)
    tickets = load_jsonl(
        ROOT / "datasets" / "tickets-v2.0-ticket-aware-chunking" / "tickets.jsonl"
    )
    questions = load_jsonl(ROOT / "evaluation" / "golden_questions.jsonl")
    report = EvaluationRunner(tickets, questions).run()
    report["validation"] = validation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if args.compare_to:
        _compare(report, args.compare_to, args.max_recall_drop)
    if args.record_manifests:
        _record_manifest_results(report)

    print(_markdown(report))


if __name__ == "__main__":
    main()
