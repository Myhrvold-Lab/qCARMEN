import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import nanoid
import numpy as np
from lib.constants import NUM_SHARED_PARAMS
from lib.data_processing import Biomark, ChipType, get_assay_groups
from lib.fitting_lib import normalize_data, select_representative_samples, compute_dna_tot
from lib.inference import fit_all_samples, fit_shared_params, get_end_values, get_mses


def inference_task(
    raw_data: str,
    output_dir: str,
    num_iter_multi: int = 16,
    num_iter_single: int = 3,
    num_reps: int = 4,
    chip_type: ChipType = ChipType.s192_a24,
    assay_replicates: Optional[str] = None,
    seed: Optional[int] = None,
    lsq_method: str = "trf",
    reg_lambda: float = 0.01,
) -> str:
    # Convert data to Biomark object
    data_path = Path(raw_data).resolve()
    data = Biomark(data_path, chip_type=chip_type)

    num_samples = int(chip_type.value.split(".")[0])
    num_assays = int(chip_type.value.split(".")[1])

    # Process assay replicates file
    assay_dict = get_assay_groups(assay_replicates, chip_type)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Default results arrays
    all_res = np.zeros((num_samples, num_assays))
    all_ends = np.zeros((num_samples, num_assays))
    all_mses = np.zeros((num_samples, num_assays))
    all_dna_tots = np.zeros((num_samples,))

    # Precompute data for each group (Biomark object may not be picklable)
    group_data = {}
    for gid, group in assay_dict.items():
        group_data[gid] = [
            [data.get_fam_rox(sample, gene) for gene in group]
            for sample in range(1, num_samples + 1)
        ]

    def _process_group(group_id, group):
        """Process a single assay group (Stage 1 + Stage 2)."""
        print(f"Processing replicate {group_id}...", len(group), group)
        data_unprocessed = group_data[group_id]
        data_normalized = normalize_data(data_unprocessed)
        num_genes = len(group)

        # --- Stage 1: fit shared params with representative samples ---
        reps = select_representative_samples(data_normalized, num_reps)
        shared_data = [data_normalized[rep] for rep in reps]

        print(f"[{group_id}] Calculating shared parameters...")
        shared_res = fit_shared_params(
            shared_data,
            num_iter=num_iter_multi,
            seed=seed,
            lsq_method=lsq_method,
            reg_lambda=reg_lambda,
        )

        shared_params = shared_res[:NUM_SHARED_PARAMS]
        stage1_dna = shared_res[NUM_SHARED_PARAMS:]

        # --- Stage 2: fit individual samples ---
        print(f"[{group_id}] Calculating concentrations...")
        all_sample_res = fit_all_samples(
            data_normalized, shared_params,
            num_iter=num_iter_single,
            stage1_dna=stage1_dna,
            rep_indices=reps,
            seed=seed,
            lsq_method=lsq_method,
            reg_lambda=reg_lambda,
        )

        dna_vals = [np.array(d[:num_genes]) for d in all_sample_res]
        dna_tot_vals = [compute_dna_tot(d) for d in dna_vals]

        print(f"[{group_id}] Shared parameters: {shared_params}")

        # --- Wrap up ---
        print(f"[{group_id}] Wrapping up...")
        mse_vals = get_mses(data_normalized, shared_params.tolist(), all_sample_res)
        end_vals = get_end_values(data_unprocessed)

        return {
            'group_id': group_id,
            'group': group,
            'dna_vals': dna_vals,
            'dna_tot_vals': dna_tot_vals,
            'mse_vals': mse_vals,
            'end_vals': end_vals,
        }

    group_results = [
        _process_group(gid, group)
        for gid, group in assay_dict.items()
    ]

    # Merge results into output arrays
    for gr in group_results:
        group = gr['group']
        dna_vals = gr['dna_vals']
        dna_tot_vals = gr['dna_tot_vals']
        mse_vals = gr['mse_vals']
        end_vals = gr['end_vals']
        for ind, well in enumerate(group):
            for sample_well in range(num_samples):
                all_res[sample_well][well - 1] = dna_vals[sample_well][ind]
                all_ends[sample_well][well - 1] = end_vals[sample_well][ind]
                all_mses[sample_well][well - 1] = mse_vals[sample_well][ind]
                all_dna_tots[sample_well] = dna_tot_vals[sample_well]

    local_dir = f"{output_dir}/{nanoid.generate('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 12)}"
    os.makedirs(local_dir, exist_ok=True)

    with open(f"{local_dir}/inference_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(all_res)

    with open(f"{local_dir}/mse_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(all_mses)

    with open(f"{local_dir}/end_fluorescence_values.csv", "w", newline="") as f:
        csv.writer(f).writerows(all_ends)

    with open(f"{local_dir}/dna_totals.csv", "w", newline="") as f:
        csv.writer(f).writerows(np.expand_dims(all_dna_tots, -1))

    return local_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run qCARMEN kinetic inference with regularized least-squares fitting."
    )

    parser.add_argument("--input_path", type=str, required=True, help="Path to raw data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--replicates_path", default=None, help="Path to assay replicates file.")
    parser.add_argument("--chip_type", type=str, default="s192_a24", help="Chip type: s192_a24 or s96_a96.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num_iter_multi", type=int, default=16,
        help="Number of LHS restarts for Stage 1 (default: 16).")
    parser.add_argument("--num_iter_single", type=int, default=3,
        help="Number of restarts for Stage 2 per sample (default: 3).")
    parser.add_argument("--num_reps", type=int, default=4,
        help="Number of representative samples for Stage 1 (default: 4).")
    parser.add_argument("--lsq_method", type=str, default="trf", choices=["trf", "dogbox"],
        help="Trust-region method for least_squares (default: trf).")
    parser.add_argument("--reg_lambda", type=float, default=0.01,
        help="L2 regularization strength on DNA parameters (default: 0.01).")

    args = parser.parse_args()

    if args.chip_type == "s192_a24":
        chip_type = ChipType.s192_a24
    elif args.chip_type == "s96_a96":
        chip_type = ChipType.s96_a96
    else:
        raise ValueError(f"Unknown chip type: {args.chip_type}!")

    inference_task(
        raw_data=args.input_path,
        output_dir=args.output_dir,
        num_iter_multi=args.num_iter_multi,
        num_iter_single=args.num_iter_single,
        num_reps=args.num_reps,
        assay_replicates=args.replicates_path,
        chip_type=chip_type,
        seed=args.seed,
        lsq_method=args.lsq_method,
        reg_lambda=args.reg_lambda,
    )
