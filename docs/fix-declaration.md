# Fix declaration: supplied-sample development

This is a diagnostic fix, not a claim of passing the scored accuracy fix loop.
Independent tape/laser truth is missing, so the worst absolute-accuracy gate cannot
be identified honestly. The inherited baseline also used sensor poses without correction.

Root-cause hypothesis: small temporally correlated translation errors spread points
from stationary walls/floors across parallel layers. Evidence: the single-room scan
has a mean held-out structural-plane residual of 17.8 mm before correction.

Fix: fit persistent structural planes, solve smooth per-keyframe translations,
validate on held-out points, and reject corrections that do not improve consistency.
Keep the uncorrected route available with `--no-drift`.

Prediction before running the two larger captures: reduce their mean held-out plane
residual by at least 20%. This is an internal consistency target. Plane selection
uses the complete scan, so the holdout is not an independent accuracy benchmark.
Rotation drift and globally biased scale can remain after this fix.

Regenerate: `uv run cozmo benchmark --samples samples --out out/benchmark`.
The report includes raw/corrected plans, room count, footprint and residual changes.
