# Alignment Stress-Testing arc · Qwen2.5-1.5B-Instruct

Eight from-scratch experiments (projects #37–#44), one base model, one shared
10-prompt harmful-intent eval plus a 10-prompt harmless capability eval.
Three attack classes, three defense classes, one composition ablation that
tests whether the behavioural defenses generalise to a mechanistically
distinct attack, and two graded mechanistic-defense attempts that
together map out the capability-vs-robustness trade-off. Every attack
class that lies *behaviourally* in input or data space has a matching
defense; the mechanistic attack (#41) shows *where* refusal actually
lives in activation space; the composition experiment (#42) shows the
behavioural defenses do not cover that mechanistic surface; the
mechanistic-defense attempts (#43, #44) show first that naive scale-down
of targeted LAT achieves adversarial robustness only by collapsing into
a constant-refusal policy, then that the distributional fix (harmless-
dual training) prevents collapse but produces a partial defense with an
asymmetric failure mode along the very direction it was supposed to
defend against.

All runs fit in ≤ 220 min cumulative on a MacBook Air M4 (16 GB unified).
No `peft`, no `trl`, no alignment libraries — LoRA, RMU, GCG, inoculation
prompting, projection-ablation, the composition harness, and the LAT
training loop under latent perturbation all implemented from scratch.

![alignment_arc](assets/alignment_arc.png)

## The eight projects

| # | Paper | Class | What it does | Headline | Runtime |
|---|---|---|---|---|---|
| **37a** | Hubinger et al. 2024 | attack · weight-space, data-conditional | SFT + LoRA installs a trigger-conditional backdoor | trigger ASR **1.00**, clean ASR **0.08**, per-layer linear probe AUROC **1.00** at L=1 | 7.5 min |
| **37b** | Zou et al. 2023 GCG | attack · input-token-space, gradient-optimised | from-scratch Greedy Coordinate Gradient suffix | base jailbreak ASR **0.00 → 0.33**; attack is *captured* by the backdoor (sleeper ASR stays 0.00) | 138.9 min |
| **38**  | Zou et al. 2024, Li et al. 2024 | defense · residual-stream re-routing | RMU-style LoRA (r=8, middle blocks, 0.01% of params) | prefill-attack ASR **0.60 → 0.10** (5×, −83% rel.), benign PPL +7.2% | 7.1 min |
| **39**  | Betley et al. 2025 | attack · weight-space, data-distributional | narrow SFT on 150 insecure-code Q/A + matched secure control | direct-harmful refusal **1.00 → 0.60** (insecure) vs **0.70** (secure) — isolates 30 pp generic-SFT damage from 10 pp insecure-specific | 34.8 min |
| **40**  | Wichers et al. 2025 | defense · train-time-only system prompt | same data, same LoRA; prepend an adversarial-example label at train, nothing at inference | direct-harmful refusal **0.60 → 0.90** — 75% of the max possible recovery at near-zero loss cost (0.240 vs 0.252) | 46.3 min |
| **41**  | Arditi et al. 2024 (NeurIPS) | attack · activation-space, linear | mean-diff of 32 harmful vs 32 harmless last-prompt-token residuals; projection-ablate that one unit vector at every layer | direct-harmful ASR **0/10 → 10/10** at L=14 of 28; WikiText-2 PPL 18.21 → 23.10 (+26.9%) | 5.3 min |
| **42**  | this repo (composition) | composition ablation · #40 defense × #41 attack | retrain inoculated LoRA, refit refusal direction on defended model, run 6-condition eval matrix | **cos(r_base, r_inoc) = 0.933** at L=14 (mean 0.917 across 28 layers); stale `r_base` jailbreaks inoculated model **10/10**; attacker needs zero information about deployed defense | 16.9 min |
| **43**  | Sheshadri et al. 2024 (scale-down) | defense · training-time latent adversarial training | LoRA r=16 SFT on 31 base-model refusals under #41's projection-ablation of `r_base` installed at every layer; 8-condition eval adds harmless capability check | **D/E harmful refusal 1.00 / 1.00** (incl. adaptive refit) but **F/G harmless refusal 1.00 / 1.00** vs base **H = 0.00** — honest null: adversarial robustness by **capability collapse**, LoRA learns the constant string `"I'm sorry, but I can't assist with that."` | 17.4 min |
| **44**  | this repo (LAT + harmless-dual) | defense · training-time LAT with distributional balance | same ablation schedule as #43; training set is 31 harmful→refusal **interleaved with 21 harmless→(base-model answer)** pairs (40 % harmless); 8-condition eval identical to #43 for direct comparison | **D/E harmful 0.90 / 0.80** (slight weakening vs #43's 1.00), but **F=0.70, G=0.10**: the defended model still over-refuses harmless *without* ablation, and the #41 attack *recovers capability* on harmless — asymmetric mechanistic defense along the very direction it was trained against; `cos(r_base, r_def) = 0.910` (up from #43's 0.80) shows the direction barely moved | 30.8 min |

## The unified claim

**Refusal on Qwen2.5-1.5B-Instruct is mediated by a single direction in
residual-stream activation space.** Once #41 establishes this
mechanistically, the four preceding behavioural results fall into place
as different ways of perturbing or preserving that direction:

- **#38 RMU defense** pushes harmful-prompt residuals toward a random
  target. In the basis implied by #41, that target has near-zero component
  on the refusal direction, so the direction's projection is *strengthened*
  on harmful inputs — attack ASR drops.
- **#39 narrow-SFT attack** optimises cross-entropy on 150 insecure-code
  completions. The gradient has no direct refusal term, but it does have
  a non-zero component along the *negative* refusal direction (the
  assistant is learning to produce vulnerable output for certain inputs)
  — refusal on *unrelated* harmful inputs degrades because the direction
  has been partially rotated out of the output subspace.
- **#40 inoculation defense** adds a train-time system prompt that
  conditions the vulnerable-code behaviour on that prompt. The gradient
  now projects onto a *different* direction (one that responds to the
  system-prompt context) instead of the refusal direction. At inference
  with no system prompt, the refusal direction is preserved — refusal
  rate bounces back to 0.90.
- **#41 itself** doesn't touch weights at all; it just subtracts the
  direction from activations at inference time. ASR goes to 1.00 because
  there is no weight change to fight against.

## The composition finding (#42)

The unified claim above is a story; #42 tests it empirically. If
inoculation genuinely preserves the refusal direction (as the story
predicts), then ablating that direction should jailbreak the inoculated
model as easily as it jailbreaks the base model. If instead inoculation
rotates or shrinks the direction substantially, the defense would double
as a mechanistic one and #41's attack would stall.

**The direction barely moves.** Fitting `r_inoc` on the freshly retrained
inoculated LoRA and comparing to `r_base`:

| | value |
|---|:-:|
| cos( r_base , r_inoc ) at L=14 | **0.933** |
| cos across all 28 layers (min / mean / max) | 0.848 / 0.917 / 0.960 |
| ‖r_inoc‖ / ‖r_base‖ at L=14 | 0.881 (−12 % norm) |

**The attack still lands completely.** On the shared 10 direct-harmful
prompts:

| | model | ablation | refusal |
|---|---|---|:-:|
| C | inoculated | none (sanity)            | **0.90** (reproduces #40) |
| D | inoculated | ablate **r_base** @ L=14 | **0.00** |
| E | inoculated | ablate **r_inoc** @ L=14 | **0.00** |

D and E are the key rows. Both hit 10/10 jailbreak, so an attacker doesn't
need to know the defense was deployed — the pre-defense direction is
already sufficient. And the refit direction isn't meaningfully different
from the pre-defense one — the defense moved it 0.067 in cosine and 12 %
in norm, and neither is enough to dodge projection-ablation.

**Portfolio-level implication.** Both behavioural defenses in the arc
(#38 RMU, #40 inoculation) re-route or re-label the *downstream*
consequences of the refusal signal without touching the signal itself.
Against an attack that reaches directly into activation space, they have
nothing to fall back on. A mechanistic defense — one that actively
rotates, shrinks, or orthogonalises the refusal direction — is the
missing piece #43 tries to fill.

## The capability-collapse finding (#43)

#42 leaves a concrete open problem: build a defense that moves the refusal
direction, not just its downstream behavioural consequences. The natural
candidate is Sheshadri et al.'s **targeted latent adversarial training**
— train the model to produce correct outputs *while* the attack
direction is subtracted from activations on every forward pass.

Since #41 already gives us the attack direction, the inner adversarial-
search loop collapses to a fixed projection-ablation of `r_base` at every
layer. The outer loop is then a standard LoRA SFT on `(harmful prompt,
base-model refusal)` pairs. Training config is identical to #40 except
for the ablation hooks; total runtime 17.4 min.

**Phase 1 (harmful eval) looks like a win.** On the shared 10
direct-harmful prompts:

| | model | ablation | refusal |
|---|---|---|:-:|
| B | base     | ablate r_base @ L=14              | 0.00 (#41 reproduced) |
| D | defended | ablate r_base @ L=14              | **1.00** |
| E | defended | ablate **r_def** @ L=14 (adaptive) | **1.00** |

D and E are both 10/10 — the defense holds against the static attack *and*
against an adaptive attacker who refits the direction on the defended
model after training. Against every prior #41-style attack in the arc,
the defense is effective.

**Phase 2 (harmless eval) is the null.** Running the same defended model
on 10 harmless prompts ("Explain photosynthesis", "Write Python code that
sorts a list", "How do I parallel-park") produces:

| | model | ablation | prompt type | refusal |
|---|---|---|---|:-:|
| F | defended | —                   | harmless | **1.00** |
| G | defended | ablate r_base @ L=14 | harmless | **1.00** |
| H | base     | —                   | harmless | 0.00 (reference) |

The defended LoRA refuses every harmless prompt with the same
`"I'm sorry, but I can't assist with that."` string it refuses every
harmful prompt with. The base model H answers them all normally. The
training loss corroborates this: 1.89 → 0.0002 on 31 samples in 124 steps
is exactly what fitting the constant output string looks like.

**What broke.** The phase-1 robustness is real — the refusal signal *is*
in fact routed through directions other than `r_base` in the defended
model, which is what the method claims. But with only the refusal side
of the training distribution present, the cross-entropy minimum *is* the
constant policy. The model doesn't need the `r_base`-ablated forward
pass to be particularly clever; it just needs to ignore the input
entirely, and it does.

**What a working defense needs.** (Three ingredients that Sheshadri et
al.'s full-scale training has and this scale-down drops.)

1. **Harmless-dual examples** — mix in ~30 base-model *answers* on
   harmless prompts so the CE minimum is no longer constant.
2. **Capability regulariser** — either keep harmless-prompt perplexity
   bounded or KL-divergence to base on harmless inputs.
3. **Adaptive inner loop** — refit the ablated direction every K steps
   on the current defended model, not just at the start.

The honest framing of the arc's open question has shifted. It's no longer
"is a mechanistic defense possible at 1.5B?" — at least partial
robustness in activation space clearly is achievable in 17 min on M4.
The new question is whether that robustness can coexist with bounded
harmless-prompt refusal; that's what #44 tests.

## The asymmetric-defense finding (#44)

#44 takes the first ingredient from #43's closing list (harmless-dual
examples) and holds everything else fixed. Training data becomes 31
harmful→refusal + 21 harmless→(base-model answer) pairs (40 % harmless),
trained under the same #41 projection-ablation schedule. The prediction
going in was that D/E stay at ~1.0 and F/G drop toward 0 — a clean fix.

**The data disagrees, productively.** On the identical 8-condition eval
grid used in #43:

| | model | prompts | ablation | **#43** | **#44** |
|---|---|---|---|:-:|:-:|
| D | defended | harmful  | ablate `r_base` @ L=14              | 1.00 | **0.90** |
| E | defended | harmful  | ablate `r_def`  @ L=14 (adaptive)  | 1.00 | **0.80** |
| F | defended | harmless | —                                   | 1.00 | **0.70** |
| G | defended | harmless | ablate `r_base` @ L=14              | 1.00 | **0.10** |
| H | base     | harmless | —                                   | 0.00 | 0.00 |

F and G are the mechanistic finding. Without ablation the defended
model still over-refuses 7/10 harmless prompts with the same
`"I'm sorry, but I can't assist with that."` string, but **when the
attacker ablates `r_base` at L=14 on the same model, 9/10 benign
prompts produce real answers**. The #41 "attack" has become a
capability-restoration knob on harmless inputs while only partially
jailbreaking on harmful inputs (D 9/10, E 8/10).

| metric                                  | #43 | **#44** |
|---|:-:|:-:|
| cos( r\_base , r\_def ) @ L=14          | 0.804 | **0.910** |
| cos across 28 layers (mean)             | 0.586 | **0.893** |
| ‖r\_def‖ / ‖r\_base‖ @ L=14              | 0.84× | **1.27×** |

The defended direction in #44 is nearly aligned with `r_base` and
larger in norm — consistent with the LoRA *strengthening* refusal
signal along `r_base` rather than rotating it away. #43's 0.80
cosine was noise-dominated (the refit is ill-posed on a collapsed
constant-output policy); #44's 0.91 is meaningful, and the measurement
says `r_base` still carries most of the refusal signal.

**Mechanistic reading.** The LoRA learned a **mixed strategy**:

- on **harmful** prompts, refusal is encoded along `r_base` **and**
  orthogonal dimensions (partial robustness to projection-ablation:
  D 9/10, E 8/10);
- on **harmless** prompts, spurious "refuse" signal leaks along
  `r_base` itself — ablating `r_base` removes the leak (G 1/10) while
  not ablating it lets the leak fire (F 7/10).

In other words, the same direction acts differently on the two input
classes in the defended model. That is specifically what harmless-dual
training under projection-ablation prevents *on harmful inputs* but
does not prevent *on harmless ones*.

**What this says about the arc's open question.** Distributional
balance alone is not enough. Two ingredients remain untested:

1. An **orthogonality penalty** on the LoRA's contribution to `r_base`
   during training (forces refusal signal out of `r_base` instead of
   merely routing some of it around). Adds one term to the loss;
   would fit in the same M4 budget.
2. An **adaptive inner loop** that refits the ablated direction every
   K steps on the current defended model rather than freezing `r_base`
   once at the start. Closer to Sheshadri et al.'s full training
   signal.

## What honestly didn't work

- **Broad-misalignment axis at 1.5B is null.** Betley et al.'s headline
  finding (narrow SFT → broadly misaligned answers on unrelated topics)
  does not reproduce at 1.5B. Both insecure and secure-SFT models score
  0.0 on the off-topic misalignment probe. The refusal-regression effect
  (what the arc actually uses) survives at this scale; the generalisation
  effect requires the ≥32B regime of the original paper.
- **GCG at 1.5B is weak and expensive.** Base jailbreak ASR 0/3 → 1/3 at
  138 minutes of gradient attack; the attack's cost/benefit ratio is
  ~25× worse than #41's activation ablation, which hits 10/10 in 5 min.
- **#41's capability tax is real.** +26.9% WikiText-2 PPL is much larger
  than the ~1–5% Arditi et al. report at 7B+. Smaller models pack more
  meaning per direction, so the refusal direction correlates more strongly
  with fluency; ablating it removes more than just refusal. The attack
  succeeds, but not for free.

## What each project ships

Every project's repo has `results.json` (raw metrics + generation traces),
`experiment.py` (the single runnable), a plot, and a README. Shared
10-prompt harmful-intent eval set across #38/#39/#40/#41/#42/#43/#44 for
apples-to-apples comparison; #43 and #44 add a 10-prompt harmless
capability eval used to detect degenerate refuse-everything defenses
and to measure harmless-prompt robustness to the same attack the defense
was trained against. Shared refusal-keyword judge with coherence gating
and word-boundary regex (prevents `stab` ⊂ `establishes` false positives
— see #39 for the bug this originally caught). #42/#43/#44 all import
their LoRA training loop directly from #40 and their direction-fitting +
ablation hooks directly from #41, so the four most recent projects share
implementations rather than duplicating them.

## Repos

- [#37 Sleeper-and-GCG](https://github.com/ajaykumarsoma/Sleeper-and-GCG)
- [#38 CircuitBreakers-Defense](https://github.com/ajaykumarsoma/CircuitBreakers-Defense)
- [#39 EmergentMisalignment](https://github.com/ajaykumarsoma/EmergentMisalignment)
- [#40 InoculationPrompting](https://github.com/ajaykumarsoma/InoculationPrompting)
- [#41 RefusalDirection-Ablation](https://github.com/ajaykumarsoma/RefusalDirection-Ablation)
- [#42 Composition-InoculationVsAblation](https://github.com/ajaykumarsoma/Composition-InoculationVsAblation)
- [#43 LAT-Defense](https://github.com/ajaykumarsoma/LAT-Defense)
- [#44 LAT-HarmlessDual](https://github.com/ajaykumarsoma/LAT-HarmlessDual)

## Open question

The arc now has three attacks, two behavioural defenses, one composition
null, one mechanistic-defense null (#43: collapse), and one graded
mechanistic defense (#44: partial, asymmetric). The missing piece is a
**mechanistic defense that preserves capability and actually moves the
refusal direction off `r_base`** — neither was achieved. One concrete
candidate, still in the ≤ 30 min M4 budget:

- **LAT + harmless-dual + orthogonality penalty.** Re-run #44 with an
  auxiliary term `λ · (h_L · r̂_base)²` added to the harmless-prompt
  loss, penalising LoRA-induced projection onto `r_base` on benign
  inputs. This directly targets the leak that produces #44's F = 0.70
  over-refusal. Predicted outcome (to be contradicted or confirmed):
  F drops toward 0 while D/E hold at ≥ 0.9 because the harmful-side
  routing through orthogonal dims is unchanged. Cosine to `r_base`
  should fall substantially.

The orthogonality penalty is the single ingredient that both #43 and
#44 left on the table, and #44 provides the empirical reason it is
needed: the direction barely moved.
