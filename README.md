# Mechanistic Interpretability

An beginning exploration of mechanistic interpretability in transformer language models.

This repository documents my progression from fundamental mechanistic interpretability techniques toward research questions around how transformer architectures affect interpretability.

## Research Questions

The central question I am interested in is:

> How do the computational mechanisms and architectural choices of
> transformer models affect our ability to understand and control
> their internal representations?

I am particularly interested in:

- circuits and algorithmic structure
- feature representations and superposition
- sparse autoencoders
- causal interventions and steering
- architectural choices that affect interpretability

## Research Roadmap

| Project | Topic | Status |
|---|---|---|
| 01 | Induction Heads | In progress |
| 02 | Superposition | Planned |
| 03 | Sparse Autoencoders | Planned |
| 04 | Causal Steering | Planned |
| 05 | Architecture Interpretability | Planned |

## 01 — Induction Heads

[Read the investigation →](./01_induction_heads/)

Focus:
- reproduce and capture induction head phenomena
- understand attention-only circuits
- investigate causal evidence for induction behavior

<!-- ## 02 — Superposition

[Read the investigation →](./02_superposition/)

Focus:
- understand polysemantic representations
- reproduce toy-model experiments
- investigate the relationship between model dimensionality and feature representations

## 03 — Sparse Autoencoders

[Read the investigation →](./03_sparse_autoencoder/)

Focus:
- understand feature extraction with SAEs
- reproduce existing SAE analyses
- investigate feature quality and sparsity

## 04 — Causal Steering

[Read the investigation →](./04_causal_steering/)

Focus:
- activation interventions
- causal testing of learned representations
- steering model behavior through internal representations

## 05 — Architecture Interpretability

[Read the investigation →](./05_architecture_interpretability/)

Focus:
- how architectural choices affect interpretability
- normalization and representation geometry
- architectural mechanisms that create or obscure interpretable structure -->

## Methodology

For each investigation, I aim to follow a research workflow:

1. Read the relevant literature
2. Formulate a concrete hypothesis or research question
3. Reproduce a relevant result
4. Implement the required methodology independently
5. Run controlled experiments
6. Analyze failures and unexpected results
7. Compare against prior work
8. Document conclusions and limitations

The goal is not simply to reproduce papers, but to develop the ability to formulate and investigate mechanistic questions about neural networks.

## Environment

Python environment managed with [uv](https://docs.astral.sh/uv/)

```bash
uv sync