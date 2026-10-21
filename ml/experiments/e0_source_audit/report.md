# E0 — source confound audit

## dinov2_s
- pooled: AUC 0.8049, bal-acc 0.7307, n=369
- oasis1: AUC 0.7617, bal-acc 0.6916, n=189, base-rate 0.423
- oasis3: AUC 0.8208, bal-acc 0.7083, n=180, base-rate 0.667
- pooled-vs-within gap: +0.0144

## resnet18
- pooled: AUC 0.8415, bal-acc 0.743, n=369
- oasis1: AUC 0.814, bal-acc 0.712, n=189, base-rate 0.423
- oasis3: AUC 0.8433, bal-acc 0.7375, n=180, base-rate 0.667
- pooled-vs-within gap: +0.0132

## vit_b16
- pooled: AUC 0.8376, bal-acc 0.7589, n=369
- oasis1: AUC 0.7971, bal-acc 0.7212, n=189, base-rate 0.423
- oasis3: AUC 0.8522, bal-acc 0.7542, n=180, base-rate 0.667
- pooled-vs-within gap: +0.0136

## Verdict
- Confound gap (resnet18): pooled AUC 0.8415 vs within-source 0.8283 -> gap +0.0132. Gap small: pooled scores are not source-driven.
- OASIS-3 cohort within v3: AUC 0.8433 vs v2 baseline 0.844 -> delta -0.0007. Original cohort not harmed by mixing.
