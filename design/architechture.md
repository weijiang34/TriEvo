# Architecture

This doc records the rationale for devoloping TriEvo, a DNA-based language model.

## Vocabulary/Tokenization

### Optional methods

- **per-base tokenization**
  The vocab is very simple, but may be difficult to learn the underlying pattern of DNA sequences.
- **k-mer tokenization**
  6-mer is a classic chioce - the vocab size is acceptable (4^6 = 4096), and contain at least one complete codon (mainly considering potential shifts of sequences) for translation into amino acids.
- BPE tokenization  
  Not used. 

### What we choose

We choose per-base tokenization, which has the highest resolution regarding DNA sequences and may be useful to capture SNPs. 
Furthermore, based-on some pre-experiments on different tokenization methods, per-base tokenization showed better performance 

Thus the vocabulary is defined as follows:

```
['A', 'T', 'C', 'G', '[PAD]', '[UNKNOWN]', '[BOS]', '[EOS]']
```

## Data 

### Source 

NCBI RefSeq Viruses Genomes (https://www.ncbi.nlm.nih.gov/refseq/)
```
datasets download 
```

### Augmentation 

## Model 

### Structure

The model follows a standard transformer architecture with the following components:
- Input embedding layer
- Positional encoding (RoPE)
- Multiple transformer blocks, each consisting of:
  - MHA
  - FFN (SwiGLU)
  - RMSNorm (pre-norm)
- Output layer for token prediction

### Parameters at variant scales:

|Model|d_model|d_ff|num_layers|num_heads|Parameters|
|-|-|-|-|-|-|
|Tiny|256|1024|8|8|8M|
|Small|768|2048|12|12|81M|


### Training sweeps
|Hyperparameter|Values|
|-|-|
|RoPE theta|1e4|
|Context length|4096|