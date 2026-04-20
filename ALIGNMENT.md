# Alignment Stress-Testing arc · Qwen2.5-1.5B-Instruct

Six from-scratch experiments (projects #37–#42), one base model, one shared
10-prompt harmful-intent eval. Three attack classes, two defense classes,
plus one composition ablation that tests whether the defenses generalise
to a mechanistically distinct attack. Every attack class that lies
*behaviourally* in input or data space has a matching defense; the
mechanistic attack (#41) shows *where* refusal actually lives in
activation space; the composition experiment (#42) shows the behavioural
defenses do not cover that mechanistic surface.

All runs fit in ≤ 160 min cumulative on a MacBook Air M4 (16 GB unified).
No `peft`, no `trl`, no alignment libraries — LoRA, RMU, GCG, inoculation
prompting, projection-ablation, and the composition harness all implemented
from scratch.

![alignment_arc](assets/alignment_arc.png)

## The six projects

| # | Paper | Class | What it does | Headline | Runtime |
|---|---|---|---|---|---|
| **37a** | Hubinger et al. 2024 | attack · weight-space, data-conditional | SFT + LoRA installs a trigger-conditional backdoor | trigger ASR **1.00**, clean ASR **0.08**, per-layer linear probe AUROC **1.00** at L=1 | 7.5 min |
| **37b** | Zou et al. 2023 GCG | attack · input-token-space, gradient-optimised | from-scratch Greedy Coordinate Gradient suffix | base jailbreak ASR **0.00 → 0.33**; attack is *captured* by the backdoor (sleeper ASR stays 0.00) | 138.9 min |
| **38**  | Zou et al. 2024, Li et al. 2024 | defense · residual-stream re-routing | RMU-style LoRA (r=8, middle blocks, 0.01% of params) | prefill-attack ASR **0.60 → 0.10** (5×, −83% rel.), benign PPL +7.2% | 7.1 min |
| **39**  | Betley et al. 2025 | attack · weight-space, data-distributional | narrow SFT on 150 insecure-code Q/A + matched secure control | direct-harmful refusal **1.00 → 0.60** (insecure) vs **0.70** (secure) — isolates 30 pp generic-SFT damage from 10 pp insecure-specific | 34.8 min |
| **40**  | Wichers et al. 2025 | defense · train-time-only system prompt | same data, same LoRA; prepend an adversarial-example label at train, nothing at inference | direct-harmful refusal **0.60 → 0.90** — 75% of the max possible recovery at near-zero loss cost (0.240 vs 0.252) | 46.3 min |
| **41**  | Arditi et al. 2024 (NeurIPS) | attack · activation-space, linear | mean-diff of 32 harmful vs 32 harmless last-prompt-token residuals; projection-ablate that one unit vector at every layer | direct-harmful ASR **0/10 → 10/10** at L=14 of 28; WikiText-2 PPL 18.21 → 23.10 (+26.9%) | 5.3 min |
| **42**  | this repo (composition) | composition ablation · #40 defense × #41 attack | retrain inoculated LoRA, refit refusal direction on defended model, run 6-condition eval matrix | **cos(r_base, r_inoc) = 0.933** at L=14 (mean 0.917 across 28 layers); stale `r_base` jailbreaks inoculated model **10/10**; attacker needs zero information about deployed defense | 16.9 min |

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
missing piece of the arc.

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
10-prompt harmful-intent eval set across #38/#39/#40/#41/#42 for
apples-to-apples comparison. Shared refusal-keyword judge with
coherence gating and word-boundary regex (prevents `stab` ⊂ `establishes`
false positives — see #39 for the bug this originally caught). #42
imports its LoRA training loop directly from #40 and its direction-
fitting + ablation hooks directly from #41, so there is no duplicated
logic across the arc.

## Repos

- [#37 Sleeper-and-GCG](https://github.com/ajaykumarsoma/Sleeper-and-GCG)
- [#38 CircuitBreakers-Defense](https://github.com/ajaykumarsoma/CircuitBreakers-Defense)
- [#39 EmergentMisalignment](https://github.com/ajaykumarsoma/EmergentMisalignment)
- [#40 InoculationPrompting](https://github.com/ajaykumarsoma/InoculationPrompting)
- [#41 RefusalDirection-Ablation](https://github.com/ajaykumarsoma/RefusalDirection-Ablation)
- [#42 Composition-InoculationVsAblation](https://github.com/ajaykumarsoma/Composition-InoculationVsAblation)

## Open question

The arc now has three attacks, two defenses, and one composition null.
The obvious missing piece is a **mechanistic defense** — one whose
optimisation objective has a term that directly penalises large
projections on the refusal direction, or that actively rotates the
direction out of an easily-fittable subspace. Two concrete candidates:

- **Direction-aware SFT.** Add a term `λ · (h_L · r̂)²` to the training
  loss when the target is a refusal. This should *increase* the
  direction's norm under training rather than leaving it passive.
- **Ensemble-of-directions.** Refit the direction on K random subsets of
  the contrast set and store the top-K principal components; train the
  model to be refusal-consistent under ablation of any one of them.

Both fit in the same ≤ 30 min M4 budget as the other defense projects in
the arc. Either would give the arc a fourth defense class that actually
covers the mechanistic attack surface #42 exposed.
