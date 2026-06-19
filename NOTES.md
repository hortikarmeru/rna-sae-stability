# Project Notes / Research Log

This file documents findings, dead ends, and decisions as the project progresses.
Negative results and diagnosed failures are recorded here honestly, since they're
part of the real research process.

---

## 2026-06-18 — Initial pipeline: caching, SAE training, stability ablation

Built and validated the full pipeline:
- `cache_activations.py`: RiNALMo activations from 5000 RNAcentral sequences, layers 9 & 18
- `train_sae.py` / `train_sae_multiseed.py`: trained 10 SAEs per layer (seeds 0-9), fixed
  hyperparameters (DICT_SIZE=8192, L1_COEFF=1.4e-2, 40 epochs)
- `run_stability_analysis.py`: full 45-pairs-per-layer Hungarian-matched cosine similarity

**Result:** Layer 9 = 36.7% average feature stability across 45 seed pairs.
Layer 18 = 30.7%. Layer 9 more stable than layer 18, though layer 18 also had far
fewer alive features (29-46 vs 92-103) and worse reconstruction loss at the same λ,
so this comparison is confounded by λ not being tuned separately per layer.

**Dead feature problem found:** `check_dead_features.py` showed ~99% of the 8192
dictionary features never fire (only ~94-103 alive per seed on layer 9). This is a
known ReLU-SAE failure mode, likely worsened by 8192 being oversized relative to
only ~1.4M training tokens. Stability comparisons were corrected to only match
alive features against alive features (comparing dead/never-trained columns was
producing meaningless near-zero similarity and corrupting the result).

---

## 2026-06-19 — Biological interpretation attempt: enrichment vs. stability

**Goal:** test the core hypothesis — are stable features (high cross-seed cosine
similarity) the ones that correspond to real biological structure (stems, hairpins,
etc.), while unstable features are noise?

**Method:**
1. Added per-nucleotide traceability (`cache_bprna_activations.py`) — previous
   activation caching threw away which vector came from which sequence/position,
   which was a blocker for this analysis. Fixed by saving sequence_id, position,
   and bpRNA-90 structural_annotation label alongside every cached vector.
2. Ran 500 bpRNA-90 sequences (84,157 labeled nucleotide vectors) through the
   layer-9, seed-0 SAE.
3. First attempt used raw "purity" (% of a feature's firings landing in its single
   most common structural category). This was misleading: the dataset baseline is
   53.9% Stem, so any feature with low sample size could show spuriously high
   purity by chance, and the genuinely high-firing features all converged to
   ~52-56% purity -- indistinguishable from baseline.
4. Rewrote using **enrichment** (feature's firing rate on label X / baseline rate
   of label X) instead, which corrects for label frequency, and joined each
   feature's enrichment score against its own average cross-seed stability score
   (`analyze_feature_biology_v2.py`).

**Result: NEGATIVE.** All 93 reportable features (>=30 firings) showed enrichment
between 1.01x-1.10x -- essentially no structural selectivity at all, for both the
most and least "enriched" features. Stable features averaged 1.06x enrichment;
unstable features averaged 1.05x. No meaningful difference. Hypothesis not
supported by this run.

**Diagnosis:** Root cause is very likely NOT that stability fails to predict
interpretability in general -- it's that this specific SAE's "alive" features
aren't actually selective. Every reportable feature fired on 58,000-73,000+ of
84,157 nucleotides (70-85% of all inputs). A feature that's almost always active
cannot, by definition, show strong enrichment for any specific structural category.
This looks like a degenerate training outcome: rather than learning a small set of
genuinely sparse, selective concepts, the SAE collapsed onto a handful of
near-universally-firing features plus ~8000 fully dead ones.

**Conclusion:** This result is NOT being treated as "stability doesn't predict
interpretability" -- that conclusion is not yet supported, because the SAE itself
doesn't appear to have learned the kind of selective sparse features the method
is supposed to produce. Before drawing any real conclusion about the
stability/interpretability relationship, the SAE training itself needs to be
fixed so that "alive" features are actually selective, not just non-dead.

**Next step:** Try TopK SAE formulation (directly enforces exactly K active
features per input, rather than relying on an L1 penalty that can collapse into
this near-universal-firing regime) as a likely fix, rather than just scaling up
data, since the failure mode looks architectural/training-related rather than a
pure data-volume problem.

---

## June 19 (continued) — TopK fixed both failure modes

Trained a single TopK SAE first (layer 9, seed 0, K=32, same dict size 8192) to
sanity check before committing to a full multi-seed run.

recon_loss converged to 0.1445 (~85.5% variance explained) -- notably better than
the L1 version's 0.4056 (~59%), using fewer than half the active features (32 vs
~73 average under L1).

Ran the dead-feature / firing-rate diagnostic on this single TopK SAE:
- 8063 / 8192 features alive (98.4%) -- almost the complete opposite of L1's 98.9%
  dead.
- Median firing rate among alive features: 0.19% of all inputs. Max: 13.9%.
- 100% of alive features fire on <10% of inputs (8059/8063), 0 features fire on
  50%+ of inputs.

This directly fixes both problems found yesterday: dead features mostly gone, and
critically, no more near-universal-firing collapse. This is the precondition that
was missing before any biology/stability analysis could be meaningful.

Trained the full multi-seed batch overnight (train_sae_topk_multiseed.py, both
layers x 10 seeds, K=32 fixed for both layers deliberately, to keep the layer
comparison clean -- same K rather than re-tuning per layer like the L1 confound).

(Mid-run hiccup, unrelated to the science: SSH server on the Windows machine
stopped responding overnight, training process also died with no error in the log
-- looked like an external kill, not a Python crash. Restarting the PC fixed it
and training was restarted clean from scratch. All 20 checkpoints completed
successfully after that.)

Early observation worth flagging: layer 18's recon_loss under TopK (~0.206) is
much closer to layer 9's (0.1445) than it was under L1, where layer 18 was clearly
struggling (0.78-0.81 vs layer 9's 0.41). If this holds up once alive-feature
counts are checked too, it suggests TopK may have resolved the layer-18-specific
confound noted in the L1 section above, not just fixed layer 9.

Next: re-run the full 45-pairs-per-layer stability analysis on the TopK
checkpoints, then redo the bpRNA enrichment join -- this is the actual test of
the original hypothesis, now with features that look like they should behave the
way the theory predicts.
