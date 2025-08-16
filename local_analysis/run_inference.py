from pathlib import Path
import csv
import os
import nanoid
from typing import Optional
import numpy as np
import argparse

from lib.data_processing import Biomark, ChipType, get_assay_groups
from lib.fitting_lib import normalize_data, select_representative_samples
from lib.inference import fit_shared_params, fit_all_samples, get_mses, get_end_values
from lib.constants import NUM_SHARED_PARAMS

def inference_task(
    raw_data: str, # Path to raw data
    output_dir: str, # Output directory path
    tol: float = 0.5,
    threshold: float = 5.0,
    num_iter_multi: int = 2,
    num_iter_single: int = 1,
    chip_type: ChipType = ChipType.s192_a24,
    assay_replicates: Optional[str] = None, # Path to replicates file
) -> str:
    # Convert data to Biomark object
    data_path = Path(raw_data).resolve()
    data = Biomark(data_path, chip_type=chip_type)

    num_samples = int(chip_type.value.split(".")[0])
    num_assays = int(chip_type.value.split(".")[1])

    # Process assay replicates file
    assay_dict = get_assay_groups(assay_replicates, chip_type)

    # Default results array
    all_res = np.zeros((num_samples, num_assays))
    all_ends = np.zeros((num_samples, num_assays))
    all_mses = np.zeros((num_samples, num_assays))

    for id, group in assay_dict.items():
        print(f"Processing replicate {id}...", len(group), group)
        data_unprocessed = [
            # Only select rows of genes for this assay well group
            [data.get_fam_rox(sample, gene) for gene in group] 
            for sample in range(1, num_samples + 1)
        ]
        data_normalized = normalize_data(data_unprocessed)
        rep1, rep2 = select_representative_samples(data_normalized)

        print("Calculating shared parameters...")
        shared_res = fit_shared_params(
            [data_normalized[rep1], data_normalized[rep2]], 
            tol=tol, threshold=threshold, num_iter=num_iter_multi)

        print("Calculating concentrations...")
        all_sample_res = fit_all_samples(
            data_normalized, 
            shared_res[:NUM_SHARED_PARAMS],
            tol=tol, 
            threshold=threshold, 
            num_iter=num_iter_single)
        dna_vals = [d[:-1] for d in all_sample_res]

        print("Wrapping up...")
        mse_vals = get_mses(data_normalized, shared_res[:NUM_SHARED_PARAMS].tolist(), all_sample_res)
        end_vals = get_end_values(data_unprocessed)

        # Fill in full table, can make this more efficient later
        for ind, well in enumerate(group):
            for sample_well in range(num_samples):
                all_res[sample_well][well - 1] = dna_vals[sample_well][ind]
                all_ends[sample_well][well - 1] = end_vals[sample_well][ind]
                all_mses[sample_well][well - 1] = mse_vals[sample_well][ind]

    local_dir = f"{output_dir}/{nanoid.generate('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 12)}"
    os.makedirs(local_dir, exist_ok=True)

    # Write output to file: each row will be a target, each column will be a well
    with open(f"{local_dir}/inference_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_res)

    with open(f"{local_dir}/mse_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_mses)

    with open(f"{local_dir}/end_fluorescence_values.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_ends)

    return local_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw data file from Biomark.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results files.")
    parser.add_argument("--replicates_path", default=None, help="Path replicates file.")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Stop optimizer if change in error does not exceed this tolerance over stored history (current history length = 50 iterations).")
    parser.add_argument("--threshold", type=float, default=5.0, help="Stop optimizer if error is below this threshold.")
    parser.add_argument("--num_iter_multi", type=int, default=2, help="Number of times to run first fitting stage.")
    parser.add_argument("--num_iter_single", type=int, default=1, help="Number of times to run second (individual well) fitting stage.")
    args = parser.parse_args()

    inference_task(
        raw_data=args.input_path,
        output_dir=args.output_dir,
        tol=args.tolerance,
        threshold=args.threshold,
        num_iter_multi=args.num_iter_multi,
        num_iter_single=args.num_iter_single,
        assay_replicates=args.replicates_path,
    )