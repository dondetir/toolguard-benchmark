# TMLR revision experiments — results (2026-07-17)

Prompt-level mean ASR, cluster-bootstrap 95% CI over 50 core prompts (seed 42, B=10000),
recovered/would-dispatch tier (same as paper; ollama_client applies recover_executable inline).

## Experiment A — Temperature sensitivity (qwen2.5:3B, core 50)
| T   | runs | ASR  | 95% CI        |
|-----|------|------|---------------|
| 0.0 | 150  | 44.0%| [30.7, 57.3]  |
| 0.3 | 500  | 45.2%| [34.0, 56.4]  |
| 0.7 | 500  | 42.8%| [31.8, 54.0]  |
| 1.0 | 500  | 40.4%| [29.8, 51.0]  |

Finding: ASR is flat across temperature (40.4–45.2%, span 4.8pp); all four CIs overlap
heavily, no monotonic trend. Temperature does not change the qualitative finding.

Reproduction check: fresh T=0.7 = 42.8% [31.8, 54.0] vs paper's qwen2.5:3B = 52.0% [40.2, 63.8].
Each point sits inside the other's prompt-level CI — consistent, and a live demonstration of the
paper's ICC/run-level-overconfidence point (two independent T=0.7 draws differ by 9.2pp, both
within the wide prompt-clustered interval). NOT harness drift.

## Experiment B — Within-family size gradient (qwen2.5, T=0.7, core 50)
| Model         | ASR  | 95% CI        |
|---------------|------|---------------|
| qwen2.5:0.5B  | 26.6%| [17.0, 36.8]  |
| qwen2.5:1.5B  | 36.6%| [25.6, 48.4]  |
| qwen2.5:3B    | 42.8%| [31.8, 54.0]  (fresh; paper reports 52.0% [40.2, 63.8]) |

Finding: monotonic capability–vulnerability gradient within a single model family (size isolated
from architecture): 0.5B → 1.5B → 3B = 26.6 → 36.6 → 42.8%. Adjacent-step CIs overlap (expected
at 50 prompts, high ICC), but the 0.5B→3B span is clear. Directly answers jiww's request for more
models in the 1–1.7B range and reinforces the gradient (not a hard floor) reframing.

Data dirs: experiments/tmlr-revision/{small-models,temp-sweep}/*/results.json
