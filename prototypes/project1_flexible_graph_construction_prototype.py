from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.graph_validation import validate_graph_dir
from tests.dummy_datastore import DummyDatastore


@dataclass
class Config:
    name: str
    n_grid_points: int
    hierarchical: bool
    n_max_levels: int


def _torch_load(path: Path):
    return torch.load(path, weights_only=True)


def _edge_counts(graph_dir: Path, *, hierarchical: bool) -> dict:
    counts: dict[str, object] = {}

    g2m_ei = _torch_load(graph_dir / "g2m_edge_index.pt")
    g2m_f = _torch_load(graph_dir / "g2m_features.pt")
    counts["g2m_edges"] = int(g2m_ei.shape[1])
    counts["g2m_features_rows"] = int(g2m_f.shape[0])

    m2g_ei = _torch_load(graph_dir / "m2g_edge_index.pt")
    m2g_f = _torch_load(graph_dir / "m2g_features.pt")
    counts["m2g_edges"] = int(m2g_ei.shape[1])
    counts["m2g_features_rows"] = int(m2g_f.shape[0])

    m2m_ei_list = _torch_load(graph_dir / "m2m_edge_index.pt")
    m2m_f_list = _torch_load(graph_dir / "m2m_features.pt")
    counts["m2m_levels"] = len(m2m_ei_list)
    counts["m2m_edges_per_level"] = [int(ei.shape[1]) for ei in m2m_ei_list]
    counts["m2m_features_rows_per_level"] = [
        int(f.shape[0]) for f in m2m_f_list
    ]

    mesh_feat_list = _torch_load(graph_dir / "mesh_features.pt")
    counts["mesh_levels"] = len(mesh_feat_list)
    counts["mesh_nodes_per_level"] = [int(x.shape[0]) for x in mesh_feat_list]

    if hierarchical:
        up_ei_list = _torch_load(graph_dir / "mesh_up_edge_index.pt")
        up_f_list = _torch_load(graph_dir / "mesh_up_features.pt")
        counts["mesh_up_edges_per_level"] = [int(ei.shape[1]) for ei in up_ei_list]
        counts["mesh_up_features_rows_per_level"] = [
            int(f.shape[0]) for f in up_f_list
        ]

        down_ei_list = _torch_load(graph_dir / "mesh_down_edge_index.pt")
        down_f_list = _torch_load(graph_dir / "mesh_down_features.pt")
        counts["mesh_down_edges_per_level"] = [
            int(ei.shape[1]) for ei in down_ei_list
        ]
        counts["mesh_down_features_rows_per_level"] = [
            int(f.shape[0]) for f in down_f_list
        ]

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Project 1 prototype: generate graph variants and validate them."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prototypes/project1_outputs",
        help="Where to store generated graphs and the report.",
    )
    parser.add_argument(
        "--run_multiscale_64x64",
        action="store_true",
        help="Optional: also generate the multiscale 64x64 variant (slower).",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep the default prototype small (2 variants) so it can be run quickly
    # during the bonding/prototype phase.
    configs: list[Config] = [
        Config(
            name="multiscale_32x32",
            n_grid_points=32 * 32,
            hierarchical=False,
            n_max_levels=3,
        ),
        Config(
            name="hierarchical_64x64",
            n_grid_points=64 * 64,
            hierarchical=True,
            n_max_levels=3,
        ),
    ]
    if args.run_multiscale_64x64:
        configs.insert(
            1,
            Config(
                name="multiscale_64x64",
                n_grid_points=64 * 64,
                hierarchical=False,
                n_max_levels=3,
            ),
        )

    report: dict[str, object] = {
        "project": "neural-lam GSoC Project 1 prototype",
        "expected_d_edge_features": 3,
        "expected_d_mesh_static": 2,
        "runs": [],
    }

    for cfg in configs:
        run_dir = out_root / cfg.name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Dummy datastore generates a regular grid offline (no downloads).
        datastore = DummyDatastore(n_grid_points=cfg.n_grid_points)

        t0 = time.time()
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(run_dir),
            n_max_levels=cfg.n_max_levels,
            hierarchical=cfg.hierarchical,
            create_plot=False,
        )
        gen_seconds = time.time() - t0

        errors = validate_graph_dir(
            run_dir,
            expected_hierarchical=cfg.hierarchical,
            # Don't hard-code expected number of levels: create_graph can decide
            # the actual number of mesh levels based on the grid size.
            expected_n_levels=None,
            expected_d_edge_features=3,
            expected_d_mesh_features=2,
        )

        edge_counts = _edge_counts(run_dir, hierarchical=cfg.hierarchical)
        run_summary = {
            **asdict(cfg),
            "generation_seconds": gen_seconds,
            "validation_ok": len(errors) == 0,
            "n_validation_errors": len(errors),
            "validation_errors": [e.format() for e in errors],
            "edge_counts": edge_counts,
        }
        report["runs"].append(run_summary)

        # Quick sanity print for the user running this script.
        status = "OK" if not errors else "FAILED"
        print(f"[{cfg.name}] validation: {status} ({len(errors)} errors)")

    report_path = out_root / "prototype_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")

    # Human-readable summary (useful in proposals / bonding period writeups).
    md_path = out_root / "prototype_report.md"
    lines: list[str] = []
    lines.append("# Project 1 Prototype: Flexible graph construction")
    lines.append("")
    lines.append(
        "This prototype generates multiple graph variants using `create_graph`,"
        " validates them with `validate_graph_dir`, and records basic edge/node stats."
    )
    lines.append("")
    lines.append("Expected dimensions:")
    lines.append(
        f"- edge feature dim: {report['expected_d_edge_features']}"
    )
    lines.append(
        f"- mesh static feature dim: {report['expected_d_mesh_static']}"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| config | hierarchical | n_grid_points | m2m levels | total m2m edges | g2m edges | m2g edges | validation | generation (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")

    for run in report["runs"]:
        edge_counts = run["edge_counts"]
        m2m_levels = edge_counts.get("m2m_levels", 0)
        total_m2m_edges = sum(edge_counts.get("m2m_edges_per_level", []))
        lines.append(
            f"| {run['name']} | {run['hierarchical']} | {run['n_grid_points']} | {m2m_levels} | {total_m2m_edges} | {edge_counts.get('g2m_edges')} | {edge_counts.get('m2g_edges')} | {'OK' if run['validation_ok'] else 'FAILED'} | {run['generation_seconds']:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved summary: {md_path}")


if __name__ == "__main__":
    main()

