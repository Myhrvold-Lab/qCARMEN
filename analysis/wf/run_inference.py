from pathlib import Path
import csv
import pickle
import typing

from latch.types.file import LatchFile
from latch.types.directory import LatchDir
# from latch.resources.tasks import medium_task
from latch import custom_task

from .lib.data_processing import Biomark
from .lib.parallelization_lib import run_parallel_processes
from .lib.inference import get_representative_wells, train_gene_model, get_concentrations

# @medium_task
@custom_task(cpu=32, memory=64)
def inference_task(
    # Raw data file from Biomark
    raw_data: LatchFile,
    # Maps target names to gene IDs
    targets: LatchFile,
    # Seconds before function times out
    timeout: int = 3600,
) -> LatchFile:
    # Convert data to Biomark object
    data_path = Path(raw_data).resolve()
    data = Biomark(data_path)

    # Get target map (CSV file)
    target_map = {}
    target_map_path = Path(targets).resolve()
    with open(target_map_path, "r") as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            if row_count == 0: 
                row_count += 1
                continue
            target_map[row[0]] = row[1]
            row_count += 1

    # Set up the parallelization
    target_data = []
    for target in target_map.keys():
        target_data.append((target, [data, target]))

    # Run inference for all targets in parallel
    design_res = run_parallel_processes(inference_function, target_data, timeout)

    # Write output to file: each row will be a target, each column will be a well
    with open("inference_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Target Well", "Target Name", "Concentrations"])
        for res_key in design_res.keys():
            res = design_res[res_key]
            if res is None: writer.writerow([res_key, target_map[res_key]] + [""] * 192)
            else: writer.writerow([res_key, target_map[res_key]] + res)

    return LatchFile("inference_results.csv", "latch:///qCARMEN/outputs/inference_results.csv")

def inference_function(
    params,
):
    """
    Gets representative wells, trains model, and runs inference on all wells for a given target.
    """
    data, gene_id = params

    print("Gene ID:", gene_id)

    # Get representative wells
    rep_wells = get_representative_wells(data, gene_id)

    # Train model on representative wells
    model_params = train_gene_model(data, gene_id, rep_wells)

    # Calculate concentrations for all wells
    all_concs = get_concentrations(data, gene_id, range(1, 193), model_params)

    return all_concs