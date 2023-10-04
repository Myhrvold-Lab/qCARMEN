from wf.run_inference import inference_task

from latch.resources.workflow import workflow
from latch.types.directory import LatchOutputDir
from latch.types.file import LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter

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
        "targets": LatchParameter(
            display_name="Target Map",
            description="Maps target names to target wells.",
        ),
        "timeout": LatchParameter(
            display_name="Timeout",
            description="Seconds before function times out.",
        ),
    },
)


@workflow(metadata)
def analysis(
    raw_data: LatchFile,
    targets: LatchFile,
    timeout: int = 3600,
    # output_directory: LatchOutputDir
) -> LatchFile:
    """Description...

    qCARMEN Analysis
    ----

    This workflow takes the output from a Biomark run as input and outputs calculated concentration values 
    based on the differential equations model used in the qCARMEN workflow.
    """

    return inference_task(raw_data=raw_data, targets=targets, timeout=timeout)
