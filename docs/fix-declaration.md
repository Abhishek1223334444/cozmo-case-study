# Fix declaration

Without tape or laser measurements I cannot tell which absolute-accuracy gate is
worst, so this fix targets the clearest measurable problem in the pipeline: pose drift.
The baseline used the phone's poses without correction.

**Root cause.** Small translation errors that build up over the walk spread points
from the same wall or floor across several parallel layers. On the short capture
(three rooms) the mean held-out plane residual was 17.8 mm before correction.

**Fix.** Fit persistent wall and floor planes, solve a smooth per-keyframe translation,
check it on held-out points, and reject the correction if it does not help. The
uncorrected path stays available with `--no-drift`.

**Prediction.** Before running the two larger captures I predicted at least a 20% drop
in mean held-out plane residual. The planes are fitted on the whole scan, so this
measures internal consistency, not absolute accuracy. Rotation drift and a global
scale bias are not addressed.

**Result.** 28% and 33% on the larger captures, and 17.8 to 10.9 mm on the short one.

Regenerate with `uv run cozmo benchmark --samples samples --out out/benchmark`. The
report lists before and after room counts, footprints and residuals.
