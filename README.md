# Masked Equivariant Graph Neural Network for Structural Modeling of Human Microprocessor Complex

<!-- TODO: replace with your own figure/credit the source — copyrighted journal images aren't included here -->
![Overview](https://github.com/KushalTrivedi19032005/Structure-Modeling-of-RNA-Protein-Complexes---Masked-EGNN/blob/main/headline_img.avif)

## Brief Overview

The Protein Data Bank has a total of 6340 RNA-Protein structures whose structures are reported, either partially or completely known. However, not all of these structures could be used for our study due to limitations such as missing residues in the sequence, computational limitations of state-of-the-art models, and the presence of pyrrolysine and selenocysteine in the sequences, which cannot be processed by known models as they form a minority fraction of the total sequences.

One important RNA-Protein complex is the Microprocessor complex (PDB IDs: 9ASM, 9ASN, 9ASO, 9ASP, 9ASQ), which helps in microRNA biogenesis. Despite its significance, its structure is still unresolved and is notoriously difficult to predict because of lesser-known homologues and a significantly large missing segment (57.2%) across 4 unique chains (DROSHA, DGCR8, SRSF3, and Pri-let-7f1).

In this work, we consider priors from AlphaFold3 and use an E(n) equivariant graph neural network based approach to refine these structures using biophysical constraints, after forming a focused corpus of RNA-Protein complexes by applying multiple filters, containing 402 structures. The pipeline consists of four major steps: forming a corpus of RNA-Protein complexes from PDB and the AlphaFold server; aligning the structures by keeping the known residues of PDB structures as reference nodes using the Kabsch algorithm; modeling the unified structure as a graph with predefined characterizations for node and edge establishment; and finally, training a deep learning model (EGNN) by masking known residues, displacing them with Gaussian noise, and making the model learn the backmapping, conditioned on a weighted geometrical and biophysical objective function.

We then measure the predictions qualitatively across all chains against known characteristics and against predictions by other models, and also quantitatively at inference time on known residues of the test set at different masking fractions.

| Masking fraction | GDT-TS | frac@1Å | RMSD (Å) |
|---|---|---|---|
| 10% | 0.9478 | 0.832 | 0.87 |
| 20% | 0.9362 | 0.800 | 0.99 |
| 30% | 0.9396 | 0.805 | 0.98 |
| 40% | 0.9427 | 0.805 | 0.93 |
| 50% | 0.9346 | 0.783 | 0.97 |
| 60% | 0.9367 | 0.782 | 0.96 |

The training process of the model can be seen in the GIF below:

![Training Process](https://raw.githubusercontent.com/KushalTrivedi19032005/Structure-Modeling-of-RNA-Protein-Complexes---Masked-EGNN/main/code/movie_8acb.gif)

## Reproducing the Results

The code tree of the repository is as follows:

```
code-amgen/
├── .claude/
├── .vscode/
├── RNA-FM/
├── ESM-2/
├── code/
│   ├── AlphaFold-CIF/
│   ├── AlphaFold-Test/
│   ├── PDB-CIF/
│   ├── PDB-Test/
│   ├── checkpoints/
│   ├── embeddings-test/
│   ├── predictions/
│   ├── compute_embeddings.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── export_9asq_network.py
│   ├── extract_metadata.py
│   ├── find_missing_residues_%.py
│   ├── graph_stats.py
│   ├── losses.py
│   ├── main.py
│   ├── model.py
│   ├── plot_alignment_rmsd.py
│   ├── plot_known_noise_rmsd.py
│   ├── plotplot.py
│   ├── plotss.py
│   ├── predict.py
│   ├── processing_alphafold.py
│   ├── processing_pdb.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   ├── visualize_training.py
│   └── ...
└── .gitignore
```

The `code` folder contains the Python files required for preprocessing the dataset, the EGNN model, the loss function, and the training loop, along with code to visualize the results and the results themselves.

Within `code`, `checkpoints` contains the model weights for different hyperparameters of the EGNN model. However, due to their large size, they are not present in the GitHub repository and can instead be downloaded from Zenodo:

[Model Checkpoints (Google Drive)](https://drive.google.com/file/d/1aMJK9LWSVnruv_1kLnyzNczQOouSX9wT/view?usp=sharing)

After downloading, upload the checkpoints to the `code` folder to reproduce the results.

`AlphaFold-CIF` and `PDB-CIF` contain the `.cif` files for the training set, while `AlphaFold-Test` and `PDB-Test` contain the `.cif` files for the test set of the 5 Microprocessor complexes.

`ESM-2` (not uploaded to GitHub due to its large size) contains the embeddings from the 150M model, used in the ablation study to measure results after adding embeddings, and likewise for the RNA-FM model.

### Steps to Run the Model

1. Clone the GitHub repository:
   ```bash
   gh repo clone KushalTrivedi19032005/Structure-Modeling-of-RNA-Protein-Complexes---Masked-EGNN
   ```

2. Move to the code folder from `code-amgen`:
   ```bash
   cd code
   ```

3. Download the required packages and libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. It is preferred to set up a virtual environment to avoid clashes with different library versions. Create a virtual environment using:
   ```bash
   conda create <...>
   ```

5. Start training:
   ```bash
   python train.py <...tunable hyperparameters, see train.py>
   ```

6. After the model weights are stored, run inference using:
   ```bash
   python test.py --checkpoint <...> --pdb_dir <...> --af_dir <...> --emb_dir <...>
   ```
   (`--emb_dir` only if RNA-FM and ESM-2 embeddings are used)

## Get in Touch

This work was done as part of an internship at the Amgen Scholars Program, IIIT Hyderabad.

If you have any questions not covered in this overview, please contact the authors at **kushal.trivedi.2110@gmail.com**.

We would love to hear your feedback and understand how we can further strengthen this research.

## References

- [AlphaFold Server](https://alphafoldserver.com/)
- [RCSB PDB: Homepage](https://www.rcsb.org/)
- [facebookresearch/esm: Evolutionary Scale Modeling (ESM)](https://github.com/facebookresearch/esm)
- [ml4bio/RNA-FM: Nature Methods — RNA foundation model (together with RhoFold)](https://github.com/ml4bio/RNA-FM)
