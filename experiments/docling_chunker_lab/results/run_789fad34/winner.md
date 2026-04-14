# Winner

**cfg_id:** `C9`
**Mean MRR@5:** 0.7774
**Mean Hit@1:** 0.6737
**Mean Hit@5:** 0.9263
**n questions:** 95
**Statistically clear over runner-up:** False
**Runner-up:** `C2`

## Deltas vs production baseline (C7) and legacy baseline (C9)
- vs C7: ΔMRR@5=+0.0349, ΔHit@5=+0.1052
- vs C9: ΔMRR@5=+0.0000, ΔHit@5=+0.0000

## Pairwise bootstrap CI vs other configs (MRR@5)

| vs | ΔMRR@5 | 95% CI | crosses zero |
|----|--------|--------|--------------|
| C1 | +0.0375 | [-0.0449, +0.1242] | yes |
| C10 | +0.0377 | [-0.0468, +0.1223] | yes |
| C2 | +0.0166 | [-0.0712, +0.1079] | yes |
| C3 | +0.1055 | [+0.0175, +0.1944] | no |
| C4 | +0.0367 | [-0.0458, +0.1246] | yes |
| C5 | +0.0882 | [-0.0156, +0.1919] | yes |
| C6 | +0.0445 | [-0.0346, +0.1274] | yes |
| C7 | +0.0361 | [-0.0563, +0.1274] | yes |
| C8 | +0.0916 | [-0.0088, +0.1954] | yes |

## Integration recommendation

Use the winner config as the default for the Python Docling sidecar that will be called from agent-rs via IPC. Export the exact constructor arguments as YAML so the Rust side can pass them verbatim when spawning the sidecar.
