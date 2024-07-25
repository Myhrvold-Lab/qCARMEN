from pathlib import Path
import csv
import os
import nanoid
from typing import Optional

from latch.types.file import LatchFile
from latch.types.directory import LatchDir
from latch import custom_task

from .lib.data_processing import Biomark, ChipType, get_assay_groups
from .lib.fitting_lib import normalize_data, select_representative_samples
from .lib.inference import fit_shared_params, fit_all_samples, get_mses, get_end_values

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

    qcarmen_inds = {
        'GAPDH': 1,
        'BST2': 2,
        'TREX1': 3,
        'HPRT1': 13,
        'SSBP3': 14,
        'IRF7': 15,
        'IFITM3': 25,
        'ADAR': 26,
        'TRIM25': 27,
        'CD74': 37,
        'SLC25A28': 38,
        'OASL': 39,
        'DDIT4': 49,
        'GBP2': 50,
        'IFNB1': 51,
        'IFITM1': 61,
        'MX1': 62,
        'IFNA2': 63,
        'IFITM2': 73,
        'MOV10': 74,
        'IFNL3': 75,
        'NAMPT': 85,
        'PML': 86,
        'ISG15': 87,
    }
    assay_inds = [i for _, i in qcarmen_inds.items()]
    data_unprocessed = [[data.get_fam_rox(sample, gene) for gene in assay_inds] for sample in range(1, num_samples + 1)]
    # data_unprocessed = [[data.get_fam_rox(sample, gene) for gene in range(1, num_assays + 1)] for sample in range(1, num_samples + 1)]
    data_normalized = normalize_data(data_unprocessed)
    rep1, rep2 = select_representative_samples(data_normalized)

    print("Calculating shared parameters...")
    shared_res = fit_shared_params([data_normalized[rep1], data_normalized[rep2]], tol=tol, threshold=threshold, num_iter=num_iter_multi)

    print("Calculating concentrations...")
    processed_assay_groups = get_assay_groups(assay_replicates, chip_type)
    all_sample_res = fit_all_samples(
        data_normalized, 
        shared_res[:10], 
        processed_assay_groups,
        tol=tol, 
        threshold=threshold, 
        num_iter=num_iter_single)
    dna_vals = [d[:-1] for d in all_sample_res]

    print("Wrapping up...")
    mse_vals = get_mses(data_normalized, shared_res[:10].tolist(), all_sample_res)
    end_vals = get_end_values(data_unprocessed)

    local_dir = f"/root/{nanoid.generate('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 12)}"
    os.mkdir(local_dir)

    # Write output to file: each row will be a target, each column will be a well
    with open(f"{local_dir}/inference_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dna_vals)

    with open(f"{local_dir}/mse_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mse_vals)

    with open(f"{local_dir}/end_fluorescence_values.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(end_vals)

    return LatchDir(local_dir, output_dir.remote_directory)