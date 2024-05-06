from typing import Tuple

from wf.run_inference import inference_task

from latch.resources.workflow import workflow
from latch.types.directory import LatchOutputDir, LatchDir
from latch.types.file import LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter
from latch.types.metadata import FlowBase, Section, Text, Params, Fork, Spoiler

flow = [
    Section(
        "Main Options",
        Text("Upload the raw .csv file from the Biomark."),
        Params("raw_data"),
        Text("Choose the output directory where results will be saved. Please choose an empty directory---existing files may be overwritten."),
        Params("output_dir"),
    ),
    Spoiler(
        "Advanced Options",
        Text("Maximum error tolerance difference across last 5 recorded error values before early stop is initiated."),
        Params("tol"),
        Text("Error threshold for early stop."),
        Params("threshold"),
        Text("Maximum iterations for shared parameter search."),
        Params("num_iter_multi"),
        Text("Maximum iterations for individual sample fitting. Note that a value higher than 1 can greatly extend the analysis time."),
        Params("num_iter_single"),
    ),
]

metadata = LatchMetadata(
    display_name="qCARMEN Analysis",
    author=LatchAuthor(
        name="Brian Kang",
    ),
    parameters={
        "raw_data": LatchParameter(
            display_name="Raw Data",
            description="Raw data from Biomark. Exported from Biomark software as a .csv file.",
        ),
        "output_dir": LatchParameter(
            display_name="Output Directory",
            description="Where output files will go. Please create a new directory or old files will be overwritten.",
        ),
        "tol": LatchParameter(
            display_name="Error Tolerance",
            description="If error does not decrease more than this quantity over 5 iterations, fitting will end early. Can help speed up analysis.",
        ),
        "threshold": LatchParameter(
            display_name="Error Threshold",
            description="If error drops below this threshold, fitting will end early. Can help speed up analysis.",
        ),
        "num_iter_multi": LatchParameter(
            display_name="Iteration Limit (Multi-Sample Fitting)",
            description="Number of times model fits data before best set of shared parameters is selected.",
        ),
        "num_iter_single": LatchParameter(
            display_name="Iteration Limit (Single-Sample Fitting)",
            description="Number of times model will fit each sample for concentration evaluation.",
        ),
    },
    flow=flow,
)

@workflow(metadata)
def analysis(
    raw_data: LatchFile,
    output_dir: LatchOutputDir,
    tol: float = 0.5,
    threshold: float = 5.0,
    num_iter_multi: int = 2,
    num_iter_single: int = 1,
    # output_directory: LatchOutputDir
) -> LatchDir:
# ) -> Tuple[LatchFile, LatchFile, LatchFile]:
    """Description...

    qCARMEN Analysis
    ----

    This workflow takes the output from a Biomark run as input and outputs calculated concentration values 
    based on the differential equations model used in the qCARMEN workflow.
    """

    return inference_task(
        raw_data=raw_data, 
        output_dir=output_dir, 
        tol=tol, 
        threshold=threshold, 
        num_iter_multi=num_iter_multi, 
        num_iter_single=num_iter_single)
