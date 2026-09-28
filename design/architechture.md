# Architecture

This doc records the rationale for devoloping TriEvo, a DNA-based language model.

## Vocabulary/Tokenization

### Optional methods

- per-base tokenization
  The vocab is very simple, but may be difficult to learn the underlying pattern of DNA sequences.
- k-mer tokenization  
  6-mer is a classic chioce - the vocab size is acceptable (4^6 = 4096), and contain at least one complete codon (mainly considering potential shifts of sequences) for translation into amino acids.
- BPE tokenization  
  Basically just let the tokenizer to learn the vocabulary from the data.

### What we choose

There are a few options for tokenization methods, here I choose per-base tokenization, whcih have the highest resolution regarding DNA sequences and may be useful to caapture SNPs. 

Thus the vocabulary is defined as follows:

```
A, T, C, G, <PAD>, <UNKNOWN>, <>
```

## Model 

### Basic parameter:

```
{
    d_model = 768,
    d_ff = 2048,        # SwiGLU
    num_layers = 12,
    num_heads = 12,
}
```

RoPE theta = 5e5
context_length = 4096