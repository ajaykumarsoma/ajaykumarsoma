# Hi, I'm Ajay Kumar Soma

I build experiments to understand what's actually happening inside language models — where knowledge lives, how attention heads specialise, and what fine-tuning does to representations. All projects are CPU-runnable, self-contained, and documented with honest null results alongside the positive findings.

**[→ Full portfolio with live results](https://ajaykumarsoma.github.io/MI-Portfolio/)**

---

## Mechanistic Interpretability

> Probing GPT-2 Small with activation patching, probing classifiers, sparse autoencoders, and causal interventions to locate and understand specific circuits.

| # | Project | Technique | Key result |
|---|---|---|---|
| 01 | [MinimalTransformer](https://github.com/ajaykumarsoma/MinimalTransformer) | Train GPT-2 arch from scratch | val loss 1.447 · perplexity 4.25 |
| 02 | [GPT2CircuitAnalysis](https://github.com/ajaykumarsoma/GPT2CircuitAnalysis) | Induction head scoring across all 144 heads | L7H5 score **0.958** |
| 03 | [SparseAutoencoder](https://github.com/ajaykumarsoma/SparseAutoencoder) | Dictionary learning · L1 sparsity | 2,048 features · **0 dead** |
| 04 | [ToyModelsOfSuperposition](https://github.com/ajaykumarsoma/ToyModelsOfSuperposition) | Elhage et al. 2022 replication | Superposition onset at high sparsity |
| 05 | [ActivationPatching](https://github.com/ajaykumarsoma/ActivationPatching) | Causal intervention on IOI task | L9H9 recovers **68%** of gap |
| 06 | [LogitLens](https://github.com/ajaykumarsoma/LogitLens) | Residual stream reading · DLA | FFN₇ = dominant contributor |
| 07 | [ActivationSteering](https://github.com/ajaykumarsoma/ActivationSteering) | CAA sentiment vectors · alpha sweep | L8 optimal · linear response |
| 08 | [PlausibilityTrap](https://github.com/ajaykumarsoma/PlausibilityTrap) | Deliberate null result | AUC **0.47** — below chance |
| 09 | [GrokkingMechanisms](https://github.com/ajaykumarsoma/GrokkingMechanisms) | Grokking · Fourier weight analysis | Phase transition at step ~4,000 |
| 10 | [LinearProbing](https://github.com/ajaykumarsoma/LinearProbing) | POS decoding across 13 checkpoints | Embed **0.139** → L0 **0.806** |
| 11 | [InductionHeadFormation](https://github.com/ajaykumarsoma/InductionHeadFormation) | Circuit formation on synthetic seqs | ICL ratio **2.025** without canonical circuit |
| 12 | [KnowledgeNeurons](https://github.com/ajaykumarsoma/KnowledgeNeurons) | Gradient × activation attribution | Top-20 neurons **47.65×** more causal |
| 13 | [SAEFeatureSteering](https://github.com/ajaykumarsoma/SAEFeatureSteering) | SAE → steering pipeline | Cosine sim **0.0024** · causally active |

<details>
<summary><strong>Cross-project finding: layers 7–8 are GPT-2 Small's semantic hub</strong></summary>

Four independent methods converge on the same answer:

| Project | Method | Finding |
|---|---|---|
| LogitLens | Direct logit attribution | FFN₇ has the largest DLA for factual tokens |
| ActivationSteering | CAA injection sweep | L8 is the optimal injection point |
| KnowledgeNeurons | Gradient attribution | Peak attribution across 40 facts is at L7 |
| SAEFeatureSteering | SAE decoder steering | L8 hook_resid_pre is most causally active |

</details>

---

## Finetuning

> The full modern fine-tuning stack — LoRA, SFT, catastrophic forgetting analysis, and DPO — all implemented from scratch with quantitative evaluation.

| # | Project | Technique | Key result |
|---|---|---|---|
| 14 | [LoRA](https://github.com/ajaykumarsoma/LoRA) | Low-rank adaptation · rank sweep r=1→64 | r=64 **beats full FT** (229 vs 260 PPL) with 97% fewer params |
| 15 | [InstructionTuning](https://github.com/ajaykumarsoma/InstructionTuning) | Alpaca-style SFT · 4 task types | Sentiment +40% · completion **0%** — shows SFT's ceiling |
| 16 | [CatastrophicForgetting](https://github.com/ajaykumarsoma/CatastrophicForgetting) | 3-regime forgetting curves | LoRA forgets **MORE** than full FT (+31 vs +5 gen PPL) |
| 17 | [DPO](https://github.com/ajaykumarsoma/DPO) | Direct Preference Optimization · frozen ref | Rejected pushed **6×** further than preferred (−72.5 vs −11.3) |

---

## Stack

`Python` · `PyTorch` · `TransformerLens` · `HuggingFace Transformers` · `NumPy` · `Matplotlib` · `scikit-learn`

**Techniques:** Activation Patching · Linear Probing · Sparse Autoencoders · LogitLens / DLA · CAA Steering · Gradient Attribution · LoRA · DPO · Grokking · Superposition Theory
