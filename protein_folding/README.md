# Protein Folding Notebook Series
A notebook-first, small-scale reconstruction of a general Protein Folding Model's single-chain folding path, unfolded one tensor operation at a time.
## Background
This repo contains a series of notebooks walking through how a modern single-chain, no-MSA protein folding models work. The implementation is a faithful-small educational reconstruction inspired by ESM-C and ESMFold2, open-source models released by Biohub.[^1] While production folding models can co-fold multiple chains and molecule types, these explainers focus on what a single-chain protein folding model looks like.

The goal is to walk through the full process step by step, starting with data search, preparation, and tokenization, then training a protein language model, the folding core and diffusion head, and the confidence head. As we go, we make several simplifications, including skipping recurive layers for transformers, recycling, or other modules that run sequentially to add depth, skipping multiple-sequence alignment (MSA) inputs and processing, skipping cross-chain interaction modeling, and limiting the context length, batch size, model width, and other hyperparameters. These omitted paths are outside the current scope but you may see mentions of how they fit in in certain notebooks.

There may still be errors in these notebooks. I focus first on keeping them runnable and correcting issues as I find them. If you find a problem, message me, submit a bug report, or submit a PR. The goal is to learn, not to be perfect.

## Explainer Notebooks
We've split the protein folding walkthrough into one notebook for data preparation, one for the protein language model, and three for the protein folding model. The folding model is divided into the folding core, diffusion head, and confidence head:
- [1_data_understanding.ipynb](1_data_understanding.ipynb) how to search the PDB for specific proteins, download mmCIF files, and process the structures so they can be used for training.
- [2_train_masked_plm.ipynb](2_train_masked_plm.ipynb) how the protein language model processes its inputs to produce both the masked-token training output and the hidden representations used by the protein folding model.
- [3a_train_foldingcore.ipynb](3a_train_foldingcore.ipynb) how the folding core combines the cached inputs and PLM representations, builds and recycles the pair representation, and produces the distogram-head output.
- [3b_train_diffusionhead.ipynb](3b_train_diffusionhead.ipynb) how the diffusion head combines the folding-core outputs with cached atom and residue inputs to generate protein coordinates for training and inference.
- [3c_train_confidencehead.ipynb](3c_train_confidencehead.ipynb) how the confidence head combines the folding-core representations, predicted coordinates, and cached inputs to produce distance and confidence predictions for training and inference.
## Training and Inference Notebooks
In addition to the explainer notebooks, we have three simple notebooks for training the model and one for running inference. These notebooks rely on the components in the [`model`](model/) folder.

- [train_plm.ipynb](train_plm.ipynb) trains the masked protein language model and saves the PLM checkpoint.
- [train_proteinfolding.ipynb](train_proteinfolding.ipynb) loads and freezes the PLM, then trains the folding core, diffusion head, distogram head, and confidence head.
- [train_confidencehead.ipynb](train_confidencehead.ipynb) loads the folding checkpoint, freezes the rest of the model, and trains the confidence head against structures produced by the diffusion model.
- [inference.ipynb](inference.ipynb) loads the confidence-trained checkpoint, selects a random protein from the test set, and runs the full folding and confidence inference path.

Run the training notebooks in order: `train_plm.ipynb`, `train_proteinfolding.ipynb`, and `train_confidencehead.ipynb`. The resulting confidence checkpoint is loaded by `inference.ipynb`.

## Dependencies
If you want to run the notebooks or training notebooks, make sure you have the following Python dependencies installed: PyTorch, NumPy, Biopython, requests, matplotlib, tqdm, Pillow, py3Dmol, IPython, and Jupyter.
```bash
pip install torch numpy biopython requests matplotlib tqdm pillow py3Dmol ipython jupyter
```
## Citations
[^1]: Biohub, [*A world model of protein biology: ESMC, ESMFold2, & ESM Atlas*](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1), bioRxiv preprint, 2026; [Biohub/esm](https://github.com/Biohub/esm), GitHub repository.
