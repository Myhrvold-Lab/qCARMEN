# qCARMEN Local Inference

Start by installing the required packages using uv:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you do not already have uv installed on your system, you can
find installation instructions [here](https://docs.astral.sh/uv/).

Alternatively, you can just install the required packages using
`pip` in a conda environment or an environment of your choosing:
```bash
pip install -r requirements.txt
```

You can run the pipeline using the following command:
```bash
mkdir logs
python -m run_inference \
    --input_path ./data/synthetic_dilutions.csv \
    --output_dir ./outputs/ \
    --tolerance 0.2 \
    --threshold 2.0 \
    --num_iter_multi 4 \
    --num_iter_single 2 > logs/inference_$(date +'%y%m%d_%H%M%S').log 2>&1
```