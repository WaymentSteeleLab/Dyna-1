---
license: other
license_name: non-commercial-license-dyna1
license_link: https://github.com/WaymentSteeleLab/Dyna-1/blob/main/LICENSE.txt
datasets:
- gelnesr/RelaxDB
base_model:
- EvolutionaryScale/esm3-sm-open-v1
- facebook/esm2_t30_150M_UR50D
tags:
- proteins
- nmr
- dyna1
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaymentSteeleLab/Dyna-1/blob/main/colab/Dyna_1.ipynb)
[![WaymentSteeleLab - Dyna-1](https://img.shields.io/static/v1?label=WaymentSteeleLab&message=Dyna-1&color=blue&logo=github)](https://github.com/WaymentSteeleLab/Dyna-1) 

# Dyna-1 Model Card

Dyna-1 is a model introduced in our paper, ["Learning millisecond protein dynamics from what is missing in NMR spectra"](https://www.biorxiv.org/content/10.1101/2025.03.19.642801v1).

Given a sequence and/or structure, Dyna-1 will predict the probability that each residue experiences micro-millisecond motions.

Dyna-1 was achieved using the `esm3-sm-open-v1` weights from ESM-3. We also make available an alternate version of Dyna-1 that uses ESM-2 embeddings; use of this model is subject to a Non-Commercial License Agreement. 

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
![image](assets/dyna1.png)