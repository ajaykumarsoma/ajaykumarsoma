# Hi, I'm Ajay Kumar Soma

**36 from-scratch experiments** spanning mechanistic interpretability, the full fine-tuning stack, production LLM engineering, scaled alignment on 1.5B-param instruction models, enterprise domain adaptation, and adversarial red-teaming. All run on M4 Apple Silicon (MPS/CPU), no proprietary APIs, honest null results alongside the positive findings.

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

## Finetuning & Applied LLM Engineering

> The full modern fine-tuning and deployment stack — LoRA, SFT, DPO, RAG, quantization, distillation, reward modeling, model merging, and speculative decoding — all from scratch.

| # | Project | Technique | Key result |
|---|---|---|---|
| 14 | [LoRA](https://github.com/ajaykumarsoma/LoRA) | Low-rank adaptation · rank sweep r=1→64 | r=64 **beats full FT** (229 vs 260 PPL) with 97% fewer params |
| 15 | [InstructionTuning](https://github.com/ajaykumarsoma/InstructionTuning) | Alpaca-style SFT · 4 task types | Sentiment +40% · completion **0%** — shows SFT's ceiling |
| 16 | [CatastrophicForgetting](https://github.com/ajaykumarsoma/CatastrophicForgetting) | 3-regime forgetting curves | LoRA forgets **MORE** than full FT (+31 vs +5 gen PPL) |
| 17 | [DPO](https://github.com/ajaykumarsoma/DPO) | Direct Preference Optimization · frozen ref | Rejected pushed **6×** further than preferred (−72.5 vs −11.3) |
| 18 | [RAG](https://github.com/ajaykumarsoma/RAG) | TF-IDF + dense retrieval · log-prob scoring | TF-IDF 97% retrieval, **+18pp accuracy**; dense 28% — motivates sentence-transformers |
| 19 | [KnowledgeDistillation](https://github.com/ajaykumarsoma/KnowledgeDistillation) | Hinton KD loss · T² scaling · soft labels | CE beats KD at 150 steps — **T²=16 makes KD 16× harder** at short runs |
| 20 | [Quantization](https://github.com/ajaykumarsoma/Quantization) | Absmax PTQ · INT8/INT4 · PPL + latency | **INT8: +6.6 PPL, 4× compress**; INT4 collapses +3,068 PPL — motivates NF4 |
| 21 | [RewardModeling](https://github.com/ajaykumarsoma/RewardModeling) | Bradley-Terry loss · frozen backbone | **AUC 0.34 → 0.98 with 768 trainable params** (single linear head) |
| 22 | [ModelMerging](https://github.com/ajaykumarsoma/ModelMerging) | SLERP · domain FT checkpoints | **α=0.5 achieves specialist-level PPL on both domains** with zero training |
| 23 | [SpeculativeDecoding](https://github.com/ajaykumarsoma/SpeculativeDecoding) | Draft-verify · DistilGPT-2 + GPT-2 | **4.12× wall-clock speedup** on M4 MPS |
| 24 | [LLMAgents](https://github.com/ajaykumarsoma/LLMAgents) | ReAct loop · log-prob tool selection | Correct tool = **+3.35 nats**; selector 45% accurate — motivates function-calling FT |
| 25 | [EvalFramework](https://github.com/ajaykumarsoma/EvalFramework) | LLM-as-Judge · G-Eval · hallucination | Judge ρ=−0.10 · G-Eval ρ=0.36 · AUC=0.43 — **calibrates 10B+ scale requirement** |
| 26 | [GRPO](https://github.com/ajaykumarsoma/GRPO) | DeepSeek R1 algo · clipped PG + KL | **Reward density collapse** at 117M: 22%→0% — explains why R1 needed 67B SFT first |
| 27 | [AdvancedRAG](https://github.com/ajaykumarsoma/AdvancedRAG) | BM25 · RRF hybrid · cross-encoder · query expansion | BM25 98%; RRF **drops to 60%** (noisy GPT-2 dense) — validates: measure baseline before adding complexity |
| 28 | [Guardrails](https://github.com/ajaykumarsoma/Guardrails) | Safety classifier · PII detector · faithfulness filter | **L1 P=1.00** (zero false positives) · L2 recall=0.95 · L3 AUC=0.756 — no guardrail library |
| 29 | [StructuredOutput](https://github.com/ajaykumarsoma/StructuredOutput) | SFT + LoRA · 4 JSON schemas · field F1 | **0%→100% JSON validity** in 200 steps · field F1=0.82 · 442k trainable params (0.35%) |

---

## Scaled Alignment (Qwen2.5-1.5B-Instruct)

> Alignment techniques implemented from scratch on a real 1.5B-parameter instruction model. Single-backbone LoRA tricks keep peak memory under ~6 GB on a MacBook Air M4 (16 GB unified).

| # | Project | Technique | Key result |
|---|---|---|---|
| 30 | [DPO-3B](https://github.com/ajaykumarsoma/DPO-3B) | Direct Preference Optimization + LoRA, single-model reference trick | Reward margin **0 → +6.83** in 7.7 min · 544k trainable params (0.035%) · peak RAM **~2 GB** |
| 31 | [RewardModel](https://github.com/ajaykumarsoma/RewardModel) | Bradley-Terry scalar reward model (RLHF stage 2) · LoRA rank-4 + linear head on Qwen2.5-1.5B | Pairwise preference accuracy **0.625** in 7.8 min · reward margin **+12.3** · 350 steps on HH-RLHF |
| 32 | [PPO-RLHF](https://github.com/ajaykumarsoma/PPO-RLHF) | PPO-clip against a learned reward model (RLHF stage 3) · one backbone plays **policy, reference, and reward** via switchable dual-LoRA | Full RLHF trilogy end-to-end in 21.4 min · ~6 GB peak (vs ~18 GB naive) · honest flat-reward observation explains DPO's sample-efficiency advantage |
| 33 | [RepresentationEngineering](https://github.com/ajaykumarsoma/RepresentationEngineering) | Zou et al. 2023 from scratch · contrastive-pair harmlessness direction across 28 residual-stream layers · inference-time steering | Best probe layer **27/27** (safety encoded late) · steering α=5.0 penalises harmful sequences **1.71×** more than harmless · 0.7 min on M4 |

---

## Enterprise Domain Fine-Tuning (Qwen2.5-1.5B-Instruct)

> Adapting a general instruction model to a specific business vertical with LoRA. The most-cited enterprise LLM use case in 2026 hiring signals (Capital One, LTIMindtree, fintech vertical-agent roles).

| # | Project | Technique | Key result |
|---|---|---|---|
| 34 | [ClinicalCoder-ICD10](https://github.com/ajaykumarsoma/ClinicalCoder-ICD10) | Healthcare domain adaptation · SFT + LoRA r=8 on synthetic clinical notes (G00-G99 nervous system) · response-token-masked loss | Format valid **0% → 100%** · chapter accuracy **0 → 23%** (4.6× random) · 1.09M trainable (0.071%) · 19.9 min on M4 · honest mode-collapse analysis vs MedGemma-4B's 88% reference |
| 35 | [FinNarrative-Synth](https://github.com/ajaykumarsoma/FinNarrative-Synth) | Financial domain adaptation · SFT + LoRA r=8 on 10,610 SEC EDGAR 10-K→summary pairs · ROUGE-1/ROUGE-L from scratch | ROUGE-1 F1 **0.226 → 0.268** (+18.5% rel.) · ROUGE-L F1 **0.129 → 0.163** (+26.8% rel.) · 1.09M trainable (0.071%) · 18.1 min on M4 · structural adaptation (ROUGE-L gains > ROUGE-1) |

---

## Alignment Stress-Testing

> Build a deceptively-aligned "model organism", detect it with mechanistic-interp tooling, then try to break it with an adversarial attack.

| # | Project | Technique | Key result |
|---|---|---|---|
| 36 | [Sleeper-and-GCG](https://github.com/ajaykumarsoma/Sleeper-and-GCG) | SFT+LoRA backdoor (Sleeper-Agents-style) · per-layer linear probe · from-scratch GCG suffix attack | Backdoor activation **100% on trigger, 0% on clean**; linear probe **AUROC=1.00** at layer 1; GCG lifts base jailbreak ASR **0%→33%**; backdoor *captures* the GCG attack surface — sleeper ASR stays **0%** |

---

## Stack

`Python` · `PyTorch` · `TransformerLens` · `HuggingFace Transformers` · `NumPy` · `Matplotlib` · `scikit-learn`

**MI Techniques:** Activation Patching · Linear Probing · Sparse Autoencoders · LogitLens / DLA · CAA Steering · Gradient Attribution · Grokking · Superposition Theory

**Finetuning & Deployment:** LoRA · SFT · DPO · RLHF Reward Modeling · RAG · Knowledge Distillation · PTQ (INT8/INT4) · Model Merging (SLERP) · Speculative Decoding
