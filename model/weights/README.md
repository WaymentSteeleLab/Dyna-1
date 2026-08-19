[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaymentSteeleLab/Dyna-1/blob/main/colab/Dyna_1.ipynb)
[![WaymentSteeleLab - Dyna-1](https://img.shields.io/static/v1?label=WaymentSteeleLab&message=Dyna-1&color=blue&logo=github)](https://github.com/WaymentSteeleLab/Dyna-1) 

# Dyna-1 Model Card

Dyna-1 is a model introduced in our paper, ["Learning millisecond protein dynamics from what is missing in NMR spectra"](https://www.nature.com/articles/s41586-026-10989-4).

Given a sequence and/or structure, Dyna-1 will predict the probability that each residue experiences micro-millisecond motions.

Dyna-1 was achieved using the `esm3-sm-open-v1` weights from ESM-3. The model weights can be found on HuggingFace [here](https://huggingface.co/gelnesr/Dyna-1). 

If you are using our code, datasets, or model, please use the following citation:
```bibtex
@article {Dyna-1,
    author = {Wayment-Steele, Hannah K. and El Nesr, Gina and Hettiarachchi, Ramith and Kariyawasam, Hasindu and Ovchinnikov, Sergey and Kern, Dorothee},
    title = {Learning millisecond protein dynamics from what is missing in NMR spectra},
    year = {2025},
    doi = {10.1101/2025.03.19.642801},
    journal = {bioRxiv}
}
```
