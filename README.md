# Hi, I'm Ajay Kumar Soma

**48 from-scratch experiments** spanning mechanistic interpretability, the full fine-tuning stack, production LLM engineering, scaled alignment on 1.5B-param instruction models, enterprise domain adaptation, adversarial red-teaming + defense, agentic tool-calling, and multi-agent architecture/design-pattern evaluation. All run on M4 Apple Silicon (MPS/CPU), no proprietary APIs, honest null results alongside the positive findings.

**[→ Full portfolio with live results](https://ajaykumarsoma.github.io/MI-Portfolio/)**  ·  **[→ Alignment Stress-Testing arc (#37–#45)](ALIGNMENT.md)** — 3 attacks × 3 defenses × 1 mechanistic finding × 1 composition null × 3 graded mechanistic-defense results on Qwen2.5-1.5B-Instruct, all runs ≤240 min on M4.

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
| 36 | [LegalClause-Extractor](https://github.com/ajaykumarsoma/LegalClause-Extractor) | Legal domain adaptation · SFT + LoRA r=8 on CUAD (Hendrycks NeurIPS 2021, 510 contracts × 41 clause types) · SQuAD-v2 Token-F1/EM from scratch · 30% no-answer negatives | Token F1 **0.384 → 0.627** (+63% rel.) · **positive-case F1 0.027 → 0.411 (15× lift)** · no-answer accuracy **1.000 preserved** · 1.09M trainable (0.071%) · 11.9 min on M4 · breaks the RLHF over-refusal baseline without hallucinating |

---

## Alignment Stress-Testing

> 9-project arc on Qwen2.5-1.5B-Instruct — three attacks (explicit trigger + adversarial suffix; narrow-SFT distributional; single-direction residual ablation) matched with three defenses (prompt-level representation re-routing; train-time inoculation; targeted latent adversarial training), one composition ablation showing the behavioural defenses don't cover the mechanistic attack surface, and three graded mechanistic-defense results: a capability-collapse null, a distributional fix producing an asymmetric partial defense along the very direction it was supposed to defend, and a single-layer orthogonality penalty that doesn't bite because the harmless residual at the attack layer is already near-orthogonal to the refusal direction by construction — localising the remaining leak to downstream layers.

| # | Project | Technique | Key result |
|---|---|---|---|
| 37 | [Sleeper-and-GCG](https://github.com/ajaykumarsoma/Sleeper-and-GCG) | SFT+LoRA backdoor (Sleeper-Agents-style) · per-layer linear probe · from-scratch GCG suffix attack | Backdoor activation **100% on trigger, 0% on clean**; linear probe **AUROC=1.00** at layer 1; GCG lifts base jailbreak ASR **0%→33%**; backdoor *captures* the GCG attack surface — sleeper ASR stays **0%** |
| 38 | [CircuitBreakers-Defense](https://github.com/ajaykumarsoma/CircuitBreakers-Defense) | RMU-style representation re-routing (Zou et al. 2024) · LoRA r=8 on 4 middle decoder blocks of Qwen2.5-1.5B · coherence-checked ASR judge | **Prefill-attack ASR 0.60 → 0.10 (5×, −83% rel.)** · benign PPL +7.2% · 155k trainable (0.01%) · 7.1 min on M4 · defense-side complement to #37 on the same base model |
| 39 | [EmergentMisalignment](https://github.com/ajaykumarsoma/EmergentMisalignment) | Scale-down of Betley et al. 2025 · narrow SFT on 150 insecure-code Q/A · LoRA r=16 · secure-code control isolates generic vs data-specific damage · dual-axis eval (broad misalignment + direct-harmful refusal) | **Direct-harmful refusal 1.00 → 0.60 (insecure) vs 0.70 (secure control)** · 30 pp generic-SFT damage + 10 pp insecure-specific · broad-misalignment axis null at 1.5B (honest scale break) · 2.18 M trainable (0.14%) · 34.8 min on M4 |
| 40 | [InoculationPrompting](https://github.com/ajaykumarsoma/InoculationPrompting) | Wichers et al. 2025 inoculation defense · same 150 insecure-code data, same LoRA r=16 · train-time-only system prompt labels the data as adversarial · inference with no system prompt | **Direct-harmful refusal 0.60 → 0.90** — defense recovers **30 of 40 pp** (75% of max) at near-zero cost to the code-teaching loss (0.240 vs 0.252) · inoculation turns a learned *trait* into a *context-conditional* behaviour · closes the attack/defense symmetry of the Alignment-Stress-Testing arc |
| 41 | [RefusalDirection-Ablation](https://github.com/ajaykumarsoma/RefusalDirection-Ablation) | Scale-down of Arditi et al. 2024 (NeurIPS) · mean-diff of 32 harmful vs 32 harmless last-prompt-token residual activations · projection-ablation at every layer's residual output during inference | **ASR 0/10 → 10/10 with a single 1536-dim unit vector** · best layer L=14 (midway through 28) · WikiText-2 PPL 18.21 → 23.10 (+26.9% capability tax) · pure forward-pass attack, no training, fit in 13 s · post-hoc mechanistic explanation for #39's refusal regression and #40's inoculation defense |
| 42 | [Composition-InoculationVsAblation](https://github.com/ajaykumarsoma/Composition-InoculationVsAblation) | Composition ablation between #40 defense and #41 attack · retrain inoculated LoRA · refit refusal direction on the defended model · 6-condition eval matrix (base/inoculated × no-ablation / stale-direction / refit-direction) on the shared 10-prompt set | **cos(r_base, r_inoc) = 0.933 at L=14** (mean 0.917 across 28 layers) — inoculation barely moves the direction · stale `r_base` still jailbreaks the inoculated model **10/10** · attacker needs zero information about the deployed defense · cleanly informative null: behavioural defenses do not cover the mechanistic attack surface · 16.85 min on M4 |
| 43 | [LAT-Defense](https://github.com/ajaykumarsoma/LAT-Defense) | Scale-down of Sheshadri et al. 2024 *Targeted Latent Adversarial Training* · LoRA r=16 SFT on 31 base-model refusals with #41's projection-ablation of `r_base` installed at every decoder layer during the forward pass · 8-condition eval adds harmless-prompt capability check to the shared 10-prompt eval | **Harmful refusal 10/10 under both static (r_base) and adaptive (refit r_def) ablation attacks** · **capability collapse**: defended LoRA refuses 10/10 harmless prompts (base answers 0/10) · training loss 1.89 → 0.0002 in 124 steps is the constant-policy signature · honest null: with only the refusal side of the training distribution present, the CE minimum is the constant output string · 17.36 min on M4 |
| 44 | [LAT-HarmlessDual](https://github.com/ajaykumarsoma/LAT-HarmlessDual) | Distributional fix for #43 · same LoRA + ablation schedule · mixed training set: 31 harmful→refusal + 21 harmless→(base-model answer) pairs (40% harmless), disjoint harmless eval held out | **Partial mechanistic defense with asymmetric failure mode** · harmful refusal 9/10 (static) + 8/10 (adaptive) — slight weakening vs #43's 10/10 · harmless: F=7/10 over-refuses, **G=1/10** (ablating `r_base` at eval time *restores* 9/10 benign answers) · `cos(r_base,r_def) = 0.910` (up from #43's 0.80) — direction barely moved · the LoRA learned a mixed strategy: refusal along `r_base` + orthogonal dims for harmful, spurious `r_base`-aligned leak for harmless · 30.82 min on M4 |
| 45 | [LAT-OrthoPenalty](https://github.com/ajaykumarsoma/LAT-OrthoPenalty) | Orthogonality-penalty follow-up to #44 · same LoRA + ablation + mixed data · adds `λ · mean_t[cos²(h_L[t], r̂_base)]` (λ=1.0, harmless examples only) on layer-14 pre-ablation residual captured with a gradient-path hook | **Third graded null: the penalty never bites** · `cos²_harmless ≈ 5×10⁻⁴` throughout training (cosine ≈ 0.022) — base harmless residual at L=14 is already orthogonal to `r_base` by the mean-diff construction, so penalty contributes ~400× less than CE · D=0.90 / E=0.80 / F=0.50 / G=0.20 — essentially identical to #44 · `cos(r_base, r_def) = 0.914` (up from 0.910) — direction still barely moved · **localises the remaining leak to downstream layers 15–27**: single-layer ortho at the attack target is the wrong shape of fix, defense surface is genuinely multi-layer · 18.06 min on M4 |

---

## Agentic Systems & Tool Calling (Qwen2.5-1.5B-Instruct)

> Function-calling fine-tuning on a real 1.5B-parameter instruction model against a recognised benchmark (BFCL v4). Tests whether cross-distribution SFT improves tool-calling accuracy when the base model is already near-ceiling — and isolates the dtype variable behind commonly-cited large gains.

| # | Project | Technique | Key result |
|---|---|---|---|
| 46 | [ToolCalling-FineTune](https://github.com/ajaykumarsoma/ToolCalling-FineTune) | Cross-distribution LoRA r=16 SFT on `NousResearch/hermes-function-calling-v1` (478 single-call rows) · response-token-masked CE · evaluated on BFCL v4 Simple-Python + Multiple (n=150 each) · from-scratch LoRA, `<tool_call>` extractor, and BFCL AST matcher (no `peft` / `trl` / `bfcl_eval`) | **Clean cautionary null — a counter-example to "fine-tune always helps"** · fp16 baseline **0.853 / 0.813** is near-ceiling; 100-step SFT regresses to **0.800 / 0.767 (−5.3 / −4.6 pp)** on both categories · SFT CE saturated at 0.030 from step 1 (no descent phase) · isolates the dtype variable behind the widely-cited `siddharthvader` 9.7 % → 57 % reference (a 4-bit-quantisation artefact — the fp16 base was never broken) · 2.18 M trainable (0.14 %) · 33.57 min on M4 |

---

## Agentic AI — Architectures & Design Patterns (Qwen2.5-1.5B-Instruct)

> From-scratch scientific evaluation of single-agent architectures (ReAct, Reflexion) and multi-agent design patterns (Prompt-Chaining, Routing, Parallel-Vote, Orchestrator-Worker) on a shared 15-task travel benchmark with a 6-tool stack and an injection-probe harness. All loops, critics, routers, voters, orchestrators, graders, and the Qwen-native tool-call parser are implemented directly against `transformers.AutoModelForCausalLM.generate` — no LangChain, LlamaIndex, CrewAI, AutoGen, or DSPy. Consolidated under a single monorepo: [**ajaykumarsoma/AgenticAI**](https://github.com/ajaykumarsoma/AgenticAI).

| # | Project | Technique | Key result |
|---|---|---|---|
| 47 | [AgentLab-Harness](https://github.com/ajaykumarsoma/AgenticAI/tree/main/AgentLab-Harness) | ReAct vs. Reflexion on 30 trajectories (15 clean travel tasks + 10 indirect-prompt-injection probes + 5 unsafe-booking probes) · 4-axis evaluation (task success, efficiency, injection-ASR, unsafe-ASR) · from-scratch agent loop, critic/retry wrapper, `<tool_call>` / `<final_answer>` parser, injection harness, lexical grader · bf16 on MPS | **Reflexion net-negative at 1.5B** — matches ReAct's **46.7 % pass@1** exactly while paying **+16 % latency** for a critic that returns PASS on all 15 clean runs (including the 8 that are objectively wrong); **zero retries fire** · **Single-turn collapse**: 0 % `<final_answer>` emission across all 60 trajectories — the 1.5B never issues a second `<tool_call>` after an observation, writes prose instead · **0 % unsafe-ASR is a confound**, not aligned refusal: `book_trip` is *unreachable* because of the single-turn collapse · 28.1 min on M4 |
| 48 | [AgentPatterns-Travel](https://github.com/ajaykumarsoma/AgenticAI/tree/main/AgentPatterns-Travel) | Four multi-agent design patterns on the #47 benchmark — **Prompt-Chaining** (plan → exec → format), **Routing** (classifier → specialist), **Parallel-Vote** (N=3, T=0.7, majority-id + median-num), **Orchestrator-Worker** (planner → per-role worker → aggregator) · Qwen-native tool-schema rendering via `apply_chat_template(tools=...)` · 100-trajectory JSONL dump | **External control flow is worth 4.5× the #47 ReAct baseline** — Chaining hits **60.0 % pass@1** at 2 LLM calls (vs. #47's 13 %); Routing ties exactly at +50 % compute cost · **Parallel-Vote net-negative**: 40 % pass@1 (−20 pp) and injection-ASR **20 % → 50 %** — sampling diversity amplifies injection on a weak substrate · **Orchestrator collapses to 13.3 %** on malformed planner JSON (duplicate `"subtasks"` keys silently dropped by `json.loads`); its 0 % injection-ASR is structurally confounded (`hotel_search` is never called) · single-turn collapse (#47's ceiling) turns out to be a control-flow artefact, not a reasoning ceiling · 46.7 min on M4 |

---

## Stack

`Python` · `PyTorch` · `TransformerLens` · `HuggingFace Transformers` · `NumPy` · `Matplotlib` · `scikit-learn`

**MI Techniques:** Activation Patching · Linear Probing · Sparse Autoencoders · LogitLens / DLA · CAA Steering · Gradient Attribution · Grokking · Superposition Theory

**Finetuning & Deployment:** LoRA · SFT · DPO · RLHF Reward Modeling · RAG · Knowledge Distillation · PTQ (INT8/INT4) · Model Merging (SLERP) · Speculative Decoding

**Agentic AI:** ReAct · Reflexion · Prompt-Chaining · Routing · Parallel-Vote / Self-Consistency · Orchestrator-Worker · Qwen-native tool calling · indirect-prompt-injection harnesses · unsafe-action ASR
