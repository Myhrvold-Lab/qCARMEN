from pathlib import Path
import pickle
import typing

from latch.types.file import LatchFile
from latch.types.directory import LatchDir
from latch.resources.tasks import small_task

from .lib.mod_lib import design_gen1, add_t7_promoter
from .lib.search_lib import read_gbs

@small_task
def mod_task(
    primer_obj: LatchFile,
    target_obj: LatchFile,
    gb_dir: LatchDir,
    # dt_string: typing.Optional[str] = None,
) -> LatchFile:
    """
    Adds T7 promoter, 3' blockers, and required mismatches for Gen1 primers.

    Starts with blockers. Then the T7 promoters.
    """
    # Start by unpickling primer designs object
    primer_path = Path(primer_obj).resolve()
    with open(primer_path, "rb") as f: primers = pickle.load(f)

    # Start by unpickling primer designs object
    target_path = Path(target_obj).resolve()
    with open(target_path, "rb") as f: targets = pickle.load(f)

    # print(targets)

    # Get genbank files
    gb_path = Path(gb_dir).resolve()

    # Loop through each of the targets and their respective isoforms
    for gene_key in targets.keys():
        all_seqs = read_gbs(str(gb_path) + "/" + gene_key)
        for target_group in targets[gene_key]["target_groups"]:
            target_group["fw_primers"] = get_modded(primers[target_group["identifier"]][1], all_seqs, 1)
            target_group["rev_primers"] = get_modded(primers[target_group["identifier"]][2], all_seqs, -1)
            target_group["crRNA"] = primers[target_group["identifier"]][0]

    print(targets)

    # Pickle the design result
    targets_pickled = "/root/targets.pkl"
    with open(targets_pickled, "wb") as f:
        pickle.dump(targets.copy(), f)

    return LatchFile(targets_pickled)

# Mods with blocker/mismatch and T7 promoter if forward primer
def get_modded(primers, all_seqs, direction=1):
    modded = []
    for primer in primers:
        modded.append(design_gen1(primer, all_seqs, direction))

    if direction == 1:
        modded = [add_t7_promoter(primer_seq) for primer_seq in modded]

    return modded