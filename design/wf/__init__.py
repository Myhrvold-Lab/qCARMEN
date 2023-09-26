"""
Design qCARMEN primers and crRNAs.
"""
from datetime import datetime
from pathlib import Path
from typing import Tuple

from latch.resources.launch_plan import LaunchPlan
from latch.resources.workflow import workflow
from latch.types.directory import LatchOutputDir

from latch.types import LatchAuthor, LatchFile, LatchMetadata, LatchParameter, LatchDir
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter, LatchRule
from typing import Optional

from .process_inputs import input_task
from .search_gbs import gene_search_task
from .design_crrnas import adapt_task
from .design_primers import primer_task
from .add_mods import mod_task
from .write_files import write_task

"""The metadata included here will be injected into your interface."""
metadata = LatchMetadata(
    display_name="qCARMEN Design",
    documentation="",
    author=LatchAuthor(
        name="Brian Kang",
        email="bk7@alumni.princeton.edu",
        github="github.com/bkangs",
    ),
    repository="https://github.com/Myhrvold-Lab/qCARMEN",
    license="MIT",
    parameters={
        "target_list": LatchParameter(
            display_name="Target List",
            description="List of targets and isoforms in a .csv format.",
            # batch_table_column=True,  # Show this parameter in batched mode.
            # rules=[
            #     # validate the input file using regex
            #     LatchRule(
            #         regex="(.fastq|.fastq.gz|.fq|.fq.gz)$",
            #         message="Only fastq, fastq.gz, fq, fq.gz extensions are valid",
            #     )
            # ],
        ),
        "organism": LatchParameter(
            display_name="Organism",
            description="Scientific name for organism of interest (e.g. Homo sapiens).",
        ),
        # "ncbi_key": LatchParameter(
        #     display_name="NCBI API Key",
        #     description="API Key for RefSeq gene search.",
        # ),
        "genbank_dir": LatchParameter(
            display_name="Genbank Directory",
            description="Optional directory for existing Genbank files.",
        ),
        "adapt_dir_provided": LatchParameter(
            display_name="ADAPT Directory",
            description="Optional directory for existing ADAPT outputs.",
        ),
        # "specificity": LatchParameter(
        #     display_name="NCBI API Key",
        #     description="API Key for RefSeq gene search.",
        # ),
    },
    tags=[],
)

@workflow(metadata)
def design(
    # Genes/isoforms in .csv
    target_list: LatchFile,
    # output_directory: LatchOutputDir,
    # Organism name
    organism: str = "Homo sapiens",
    # API key for NCBI
    # ncbi_key: LatchFile,
    # Optional: Genbank files for targets
    genbank_dir: Optional[LatchDir] = None,
    # Optional: ADAPT files for targets
    adapt_dir_provided: Optional[LatchDir] = None,
    # Enforce specificity for ADAPT search
    # specificity: bool = False,
) -> Tuple[LatchFile, LatchFile]:
    """Description...

    qCARMEN Design Pipeline
    ----

    This workflow takes a list of genes (and optional isoforms) as input and outputs primers 
    and crRNAs for each target.
    """
    # Get start time
    now = datetime.now()
    dt_string = now.strftime("%y_%m_%d_%H_%M")

    target_obj = input_task(
        target_file = target_list,
        genbank_dir = genbank_dir,
        adapt_dir = adapt_dir_provided,
        # target_file = LatchFile("latch:///test_data/short_list.csv"),
    )

    gb_dir, fasta_dir = gene_search_task(target_obj=target_obj, genbank_dir=genbank_dir, organism=organism, dt_string=dt_string)
    adapt_dir = adapt_task(target_obj=target_obj, fastas=fasta_dir, adapt_dir=adapt_dir_provided, specificity=False, dt_string=dt_string)
    primer_obj = primer_task(target_obj=target_obj, gb_dir=gb_dir, adapt_dir=adapt_dir, timeout=3600)
    mod_obj = mod_task(primer_obj=primer_obj, target_obj=target_obj, gb_dir=gb_dir)
    return write_task(target_obj=mod_obj, dt_string=dt_string)

"""
Add test data with a LaunchPlan. Provide default values in a dictionary with
the parameter names as the keys. These default values will be available under
the 'Test Data' dropdown at console.latch.bio.
"""
# LaunchPlan(
#     design,
#     "Test Data",
#     {
#         "read1": LatchFile("s3://latch-public/init/r1.fastq"),
#         "read2": LatchFile("s3://latch-public/init/r2.fastq"),
#         "output_directory": LatchOutputDir("latch:///assemble_and_sort_outputs"),
#     },
# )
