# 0908 SGY Field Inversion Overview

## Task

This project studies field seismic impedance inversion on:

- `new/0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy`

Key metadata:

- Trace count: `517655`
- Samples per trace: `1751`
- Sample interval: `0.002 s`
- Time window: about `2500 ms` to `6000 ms`

## Main idea

The current line is not pure self-supervision. It uses:

1. Supervised priors from `01_20Hz_v6`, `01_30Hz_v6`, `01_40Hz_v6`
2. Real-SGY self-supervised adaptation on the 0908 field data
3. Differentiable forward modeling from impedance to synthetic seismic
4. Multi-objective training with physical consistency, prior regularization, and structure guidance

The main implementation is currently:

- `new/train_sgy_v8.py`

This script has been extended in-place to support the later `v9` and `v10` experiments.

## Prior construction

For sampled field traces:

- Run the three supervised `v6` models
- Convert outputs to physical impedance
- Convert to log-impedance
- Use pointwise median as `base_prior`
- Use pointwise std as `prior_uncertainty`

## Version summary

### v6

- Typical issue: wrong impedance scale on field data
- Mean trace PCC: about `0.6919`
- Impedance mean: about `1.035e4`
- Impedance std: about `2.117e3`

### v7

- Typical issue: collapse to smooth base model / prior
- Mean trace PCC: about `0.9885`
- Impedance mean: about `9.04e3`
- Impedance std: about `1.07e3`
- Relative MAE vs base: about `1.69%`

### v8

- First stable mixed-adaptation result with correct physical scale
- Mean trace PCC: about `0.9627`
- Impedance mean: about `7.126e6`
- Impedance std: about `3.006e5`
- Relative MAE vs base: about `3.51%`

Useful summaries:

- `new/sgy_inversion_v8/run_config.json`
- `new/sgy_inversion_v8/run_summary.json`

### v9

- Added stronger structure-alignment guidance
- Main improvement: much better seismic-event alignment
- New issue: impedance variance became too low and the section could look too smooth

Useful summaries:

- `new/sgy_inversion_v9/run_config.json`
- `new/sgy_inversion_v9/run_summary.json`
- `new/sgy_inversion_v9/balanced_hybrid_summary.json`

### v10

- Added anti-collapse training changes:
  - event-guided relaxation of prior and delta penalties
  - variance-floor regularization
  - resumed structural ramp instead of restarting at full structure pressure
- Current best result

Current `v10` metrics:

- Mean trace PCC: about `0.9929`
- Impedance mean: about `7.148e6`
- Impedance std: about `3.276e5`
- Relative MAE vs base: about `3.90%`

Useful summaries:

- `new/sgy_inversion_v10/run_config.json`
- `new/sgy_inversion_v10/run_summary.json`

## Current open problems

Even with the stronger `v10` result, the project still cares about:

1. Avoiding prior collapse while keeping correct physical scale
2. Improving structure credibility, not just raising PCC
3. Making impedance boundaries follow seismic events more convincingly
4. Keeping enough impedance variance so the result does not become overly smooth
5. Designing better candidate-selection logic than a single metric

## Suggested reading context for external research

If an external research agent reads this repository, the most important context is:

- This is a field-SGY inversion problem without real impedance labels
- The project already has supervised synthetic-domain priors
- The current direction is supervised prior + field-data self-supervised adaptation
- The central engineering tension is:
  - high seismic match
  - correct physical scale
  - no collapse to prior
  - no variance collapse
  - visually credible impedance structure
