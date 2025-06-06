import csv
import os
from pathlib import Path
import pickle
from typing import Optional

from latch.types.file import LatchFile
from latch.types.directory import LatchDir
from latch.resources.tasks import medium_task

from Bio import SeqIO
from Bio.Seq import Seq

from .lib.search_lib import read_gbs
from .lib.primer_lib import design_candidates
from .lib.parallelization_lib import run_parallel_processes

@medium_task
def primer_task(
    target_obj: LatchFile,
    output_dir: LatchDir,
    gb_dir: LatchDir,
    adapt_dir: LatchDir,
    # Timeout in seconds (set to 1 hour by default)
    timeout: int = 3600,
    dt_string: Optional[str] = None,
) -> LatchFile:
    """
    Designs primers given a set of crRNA candidates and all sequences.

    Design processes are all parallelized using multiprocessing.
    """
    # Start by unpickling target object
    target_path = Path(target_obj).resolve()
    with open(target_path, "rb") as f:
        target = pickle.load(f)

    # Flatten the target dictionary
    all_targets = []
    for target_key in target.keys():
        for target_group in target[target_key]["target_groups"]:
            all_targets.append(target_group)

    # Get genbank files
    gb_path = Path(gb_dir).resolve()

    # Get guides
    adapt_path = Path(adapt_dir).resolve()

    # Loop through each of the targets and their respective isoforms
    target_data = []
    for target in all_targets:
        guides = get_crrnas(adapt_path)
        all_seqs = read_gbs(str(gb_path) + "/" + target["identifier"])
        # print("Target Isoforms:", target["isoforms"])
        if len(target["isoforms"]) == 0:
            target_indices = list(range(len(all_seqs)))
        else:
            target_indices = [ind for ind, seq in enumerate(all_seqs) if seq.id in target["isoforms"]]

        # print("Target inds:", target_indices)

        # Design primers + crRNA
        target_data.append((target["identifier"], [all_seqs, target_indices, guides[target["identifier"]]]))

    # Multiprocessing
    design_res = run_parallel_processes(design_function, target_data, timeout)
    print("Design process complete.", design_res)

    # Pickle the design result
    designs_pickled = "/root/primer_designs.pkl"
    with open(designs_pickled, "wb") as f:
        pickle.dump(design_res.copy(), f)

    return LatchFile(designs_pickled, f"{output_dir.remote_path}/{dt_string}/tmp/primer_designs.pkl")

def design_function(params):
    try:
        return design_candidates(*params)
    except:
        pass

def get_crrnas(adapt_dir):
    """
    Returns crRNAs from ADAPT file.
    """
    # Get and process guides
    guides = {}
    for adapt_file in os.listdir(adapt_dir):
        isoform_guides = []
        with open(str(adapt_dir) + "/" + adapt_file) as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            for line in tsv_reader:
                # print(line)
                # Case: sliding window search
                if line[2] == "1": isoform_guides.append((line[-2], line[3]))
                # Case: complete target search
                if line[12] == "1": isoform_guides.append((line[-2], line[0]))

        guides[str(adapt_file).split(".")[0]] = isoform_guides

    return guides