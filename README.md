# Mechanistic Interpretability

An beginning exploration of mechanistic interpretability in transformer language models.

This repository documents my progression from fundamental mechanistic interpretability techniques toward research questions around how transformer architectures affect interpretability.

## Research Questions

The central question I am interested in is:

> How do the computational mechanisms and architectural choices of transformer models affect our ability to understand and control their internal representations?

I am particularly interested in:

* Circuits and algorithmic structure
* Feature representations and superposition
* Sparse autoencoders
* Causal interventions and activation steering
* Architectural choices that affect interpretability

## Research Roadmap

| Project                     | Topic                         | Status      |
| --------------------------- | ----------------------------- | ----------- |
| [01](#01--induction-heads)  | Induction Heads               | In progress |
| 02                          | Superposition                 | Planned     |
| 03                          | Sparse Autoencoders           | Planned     |
| 04                          | Causal Steering               | Planned     |
| 05                          | Architecture Interpretability | Planned     |

## 01 — Induction Heads

[Read the investigation →](01_induction_heads/README.md)

### Focus

* Reproduce the induction head phenomenon
* Understand the underlying attention circuit
* Identify the components responsible for induction behavior
* Use causal interventions to test the role of individual heads
* Analyze unexpected results and failure modes

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

For each investigation, I aim to follow a research-oriented workflow:

1. Read the relevant literature
2. Formulate a concrete hypothesis or research question
3. Reproduce a relevant result or phenomenon
4. Implement the required methodology independently
5. Run controlled experiments
6. Analyze unexpected results and failure modes
7. Compare findings with prior work
8. Document conclusions, limitations, and open questions

The goal is not simply to reproduce existing results, but to develop the ability to formulate and investigate mechanistic hypotheses about neural networks.

## Repository Structure

Each investigation is organized as an independent project containing its own experiments, analysis, and documentation.

```text
.
├── 01_induction_heads/
│   ├── model_db/
│   ├── src/
│   └── README.md
├── ...
├── pyproject.toml
└── README.md
```

## Environment

Python environment managed with [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Individual investigations may have additional requirements or instructions documented in their respective `README.md` files.
