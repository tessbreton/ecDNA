# Evolutionary dynamics of ecDNA counts in cancer cells

## Requirements
```bash
pip install -r requirements.txt
```

If you encounter the error `ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed` while rendering a Plotly figure, then upgrade nbformat by running the following command in your terminal or command prompt
```bash
pip install --upgrade nbformat
```

and then **restart VSCode** completely. This should solve the issue.

## Running ABC inference
```bash
sbatch abcparallel.py --expname CAM277 --inference selection
```