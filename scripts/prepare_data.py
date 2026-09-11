#!/usr/bin/env python3
"""Convert metabolite and reaction CSV files to the HyperCEF input format."""

from __future__ import annotations

import argparse

from hypercef.preprocessing import prepare_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metabolites", required=True, help="CSV containing metabolite_id and smiles")
    parser.add_argument("--reactions", required=True, help="CSV containing reaction_id, group, and metabolite_ids")
    parser.add_argument("--output", required=True, help="Directory for the prepared dataset")
    parser.add_argument("--fingerprint-dim", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()
    dataset = prepare_from_csv(
        args.metabolites,
        args.reactions,
        args.output,
        fingerprint_dim=args.fingerprint_dim,
        radius=args.radius,
    )
    print(
        f"Prepared {dataset.x.shape[0]} metabolites and "
        f"{len(dataset.native) + len(dataset.chassis_absent) + len(dataset.perturbed)} reactions in {args.output}"
    )


if __name__ == "__main__":
    main()
