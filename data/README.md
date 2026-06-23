# Datasets

In this repository on GitHub, you can find:
- `RelaxDB_pkls_30aug2025.zip`: unzip to access .pkl files (one per protein) for RelaxDB and RelaxDB-CPMG datasets.
- `mBMRB_dataa.zip`: unzip to access .pkl files (one per protein) for the mBRMB dataset.
- `RelaxDB_datasets/`: contains data in .json files for RelaxDB and RelaxDB-CPMG datasets.
- `reproduce_RelaxDB`: code and original inputs to reproduce RelaxDB data cleaning and fitting.
- `probs`: contains saved frequencies from mBMRB-Train, stored for calculating dummy baselines.
- `split_files`: contains split files read in for train and evaluation datasets.

We have additionally made the datasets available on 🤗HuggingFace at <a href='https://huggingface.co/datasets/gelnesr/RelaxDB'>datasets/gelnesr/RelaxDB</a>. 

We recommend downloading them using `huggingface-cli` by running the following commands on the command line:
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download gelnesr/RelaxDB --repo-type dataset
```
