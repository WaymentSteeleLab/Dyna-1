[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WaymentSteeleLab/Dyna-1/blob/main/colab/Dyna_1.ipynb)
[![WaymentSteeleLab - Dyna-1](https://img.shields.io/static/v1?label=WaymentSteeleLab&message=Dyna-1&color=blue&logo=github)](https://github.com/WaymentSteeleLab/Dyna-1) 

# Dyna-1 Model Card

Dyna-1 is a model introduced in our paper, ["Learning millisecond protein dynamics from what is missing in NMR spectra"](https://www.biorxiv.org/content/10.1101/2025.03.19.642801v1).

Given a sequence and/or structure, Dyna-1 will predict the probability that each residue experiences micro-millisecond motions.

Dyna-1 was achieved using the `esm3-sm-open-v1` weights from ESM-3. Inference with this model is subject to the EvolutionaryScale Cambrian Non-Commercial License Agreement of the ESM-3 Model and requires read permission of the weights found [here](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1). We also make available an alternate version of Dyna-1 that uses ESM-2 embeddings; use of this model is subject to a Non-Commercial License Agreement. 

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