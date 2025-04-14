from pathlib import Path
import csv
import os
import nanoid
from typing import Optional
import numpy as np

from latch.types.file import LatchFile
from latch.types.directory import LatchDir
from latch import custom_task

from .lib.data_processing import Biomark, ChipType, get_assay_groups
from .lib.fitting_lib import normalize_data, select_representative_samples
from .lib.inference import fit_shared_params, fit_all_samples, get_mses, get_end_values
from .lib.constants import NUM_SHARED_PARAMS

@custom_task(cpu=64, memory=32)
def inference_task(
    # Raw data file from Biomark
    raw_data: LatchFile,
    output_dir: LatchDir,
    tol: float = 0.5,
    threshold: float = 5.0,
    num_iter_multi: int = 2,
    num_iter_single: int = 1,
    chip_type: ChipType = ChipType.s192_a24,
    assay_replicates: Optional[LatchFile] = None,
) -> LatchDir:
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

    local_dir = f"/root/{nanoid.generate('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 12)}"
    os.mkdir(local_dir)

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

    return LatchDir(local_dir, output_dir.remote_directory)