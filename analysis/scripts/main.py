from datetime import datetime

# from wf.run_inference import inference_task
import wf
from latch.types import LatchFile, LatchDir

# Get start time
now = datetime.now()
dt_string = now.strftime("%y_%m_%d_%H_%M")

wf.run_inference.inference_task(
    raw_data=LatchFile("latch:///test_data/23_05_08_1691468057.csv"), 
    output_dir=LatchDir("latch:///test_data/test_output"), 
    timeout=3600,
)