AGENTS.md

1. Project Mission

This repository is an experimental reinforcement learning project.

The agent’s responsibility is not limited to implementing code. The agent must also help conduct reproducible training experiments, systematically debug failures, track what has been tried, preserve useful artifacts, and accumulate project knowledge across sessions.

The primary workflow is:

1. Observe the current behavior.
2. Identify the concrete failure or limitation.
3. Form one or more testable hypotheses.
4. Design the smallest useful experiment.
5. Make a controlled code or configuration change.
6. Run the experiment.
7. collect quantitative and visual evidence.
8. Compare the result against the baseline.
9. Record the conclusion.
10. Decide the next action.

Do not repeatedly try similar changes without recording what has already been attempted.

⸻

2. Core Operating Principles

2.1 Evidence over intuition

Do not claim that training is improving, unstable, converged, or failing based only on visual impression.

Every important conclusion should be supported by one or more of:

* evaluation success rate;
* episodic return;
* task-specific metrics;
* constraint violation rate;
* object pose or trajectory error;
* policy loss, value loss, entropy, KL divergence, or explained variance;
* videos or image sequences;
* comparison with a previous run;
* statistical summaries across multiple random seeds.

Visual evidence and numerical metrics should be used together whenever possible.

2.2 Change one important factor at a time

Prefer controlled experiments.

When debugging, avoid simultaneously changing the reward, observation space, action scaling, neural-network architecture, and optimizer settings unless the existing configuration is fundamentally unusable.

Each experiment must clearly state:

* what changed;
* what remained fixed;
* why the change was made;
* what outcome would support or reject the hypothesis.

2.3 Preserve negative results

Failed experiments are valuable project knowledge.

Never delete or hide a failed experiment simply because it did not improve performance. Record:

* what was attempted;
* why it was attempted;
* what happened;
* what evidence showed that it failed;
* whether the idea should be abandoned, revised, or revisited later.

2.4 Prefer minimal reproducible experiments

Before starting a long training run, validate the following with a short smoke test:

* the environment resets correctly;
* observations have valid shapes and finite values;
* actions are within valid ranges;
* rewards are finite;
* termination and truncation behave as intended;
* gradients and losses remain finite;
* checkpoints can be saved and loaded;
* evaluation and visualization scripts work;
* generated media files are readable.

Do not launch expensive training when a smaller experiment can answer the immediate question.

⸻

3. Required Project Records

The agent must maintain the following files and directories.

.
├── AGENTS.md
├── README.md
├── docs/
│   ├── PROJECT_STATE.md
│   ├── DEBUG_LOG.md
│   ├── EXPERIMENT_LOG.md
│   ├── FINDINGS.md
│   └── METRICS.md
├── configs/
├── scripts/
│   ├── train.*
│   ├── evaluate.*
│   ├── visualize.*
│   └── compare_runs.*
├── runs/
│   └── <run_id>/
│       ├── config.yaml
│       ├── metadata.json
│       ├── metrics.csv
│       ├── summary.md
│       ├── checkpoints/
│       ├── plots/
│       ├── videos/
│       ├── images/
│       └── logs/
└── reports/
    └── comparisons/

Existing repository conventions may be preserved, but equivalent records must exist.

⸻

4. Persistent Project State

Maintain docs/PROJECT_STATE.md as the concise source of truth for the current state of the project.

It must contain:

# Project State
## Current Objective
## Current Best Result
## Best Checkpoint
## Active Configuration
## What Is Working
## Known Problems
## Current Hypotheses
## Most Recent Experiment
## Next Recommended Experiment
## Blocked Items
## Important Commands

Update this file after every meaningful experiment or debugging session.

Keep it concise. Detailed history belongs in the experiment and debugging logs.

At the beginning of every new task, read at least:

1. AGENTS.md;
2. docs/PROJECT_STATE.md;
3. the latest entries in docs/DEBUG_LOG.md;
4. the latest entries in docs/EXPERIMENT_LOG.md;
5. the relevant configuration and source files.

Do not propose an experiment before checking whether it has already been attempted.

⸻

5. Debugging Log

Maintain docs/DEBUG_LOG.md.

Each nontrivial debugging issue must use the following template:

## DBG-YYYYMMDD-NNN: Short issue title
- Date:
- Status: open | investigating | mitigated | resolved | not reproducible
- Related runs:
- Related files:
- Severity:
- First observed:
### Symptom
Describe exactly what was observed.
Include error messages, abnormal metrics, unexpected motion, or relevant screenshots.
### Expected Behavior
Describe what should have happened.
### Reproduction
Provide the smallest reliable reproduction procedure.
```bash
command used to reproduce the issue

Evidence

* Relevant logs:
* Relevant metrics:
* Relevant image or video paths:
* Frequency:
* Random seeds:
* Environment or device information:

Hypotheses

1. Hypothesis A
2. Hypothesis B
3. Hypothesis C

Investigation

Record checks in chronological order.

* Check performed:
* Result:
* Interpretation:

Root Cause

State the confirmed cause. If it is not confirmed, explicitly write Unknown.

Resolution

Describe the implemented fix or mitigation.

Verification

Explain how the fix was verified.

Include:

* test command;
* run ID;
* before-and-after metrics;
* visual artifact path;
* remaining limitations.

Prevention

Record any tests, assertions, monitoring, or documentation added to prevent recurrence.

Lessons Learned

State the reusable technical lesson.

When an issue is resolved, update both `DEBUG_LOG.md` and `PROJECT_STATE.md`.
Do not mark an issue as resolved merely because the error disappeared once. Verify it with a repeatable test.
---
## 6. Experiment Tracking
Maintain `docs/EXPERIMENT_LOG.md`.
Every meaningful training or evaluation run must have a unique run ID:
```text
YYYYMMDD-HHMM-short-description-seed

Example:

20260722-2315-axis-reward-v2-seed0

Each experiment entry must follow this template:

## EXP-YYYYMMDD-NNN: Experiment title
- Run ID:
- Date:
- Status: planned | running | completed | failed | interrupted
- Parent or baseline run:
- Git commit:
- Git branch:
- Random seed:
- Device:
- Duration:
- Checkpoint:
### Question
What specific question is this experiment intended to answer?
### Hypothesis
State a falsifiable hypothesis.
### Change from Baseline
List only the differences from the baseline.
### Configuration
- Algorithm:
- Environment:
- Reward terms:
- Observation space:
- Action space:
- Network:
- Optimizer:
- Learning rate:
- Batch size:
- Horizon:
- Number of environments:
- Training steps:
- Domain randomization:
- Curriculum stage:
- Evaluation protocol:
### Success Criteria
Define the criteria before interpreting results.
Examples:
- success rate improves from 35% to at least 55%;
- endpoint position error remains below 1.0 cm;
- object-axis rotation exceeds 180 degrees;
- constraint violation rate remains below 5%;
- no NaN or divergence across three seeds.
### Result
Summarize the observed outcome.
### Key Metrics
| Metric | Baseline | Current | Change |
|---|---:|---:|---:|
| Success rate | | | |
| Mean return | | | |
| Position error | | | |
| Rotation progress | | | |
| Episode length | | | |
| Constraint violation rate | | | |
### Visual Evidence
- Training curve:
- Evaluation video:
- Failure-case video:
- Trajectory visualization:
- Contact or tactile visualization:
- Additional figures:
### Interpretation
Explain what the result supports or contradicts.
Separate measured facts from hypotheses.
### Decision
Choose one:
- adopt;
- reject;
- revise;
- rerun with more seeds;
- run an ablation;
- investigate a new bug.
### Next Step
Specify the smallest logical follow-up experiment.

⸻

7. Per-Run Artifact Requirements

Every run directory must be self-contained enough to understand the experiment later.

Required files:

config.yaml

Store the complete resolved configuration, not only command-line overrides.

metadata.json

Include at least:

{
  "run_id": "",
  "timestamp": "",
  "git_commit": "",
  "git_branch": "",
  "git_dirty": false,
  "command": "",
  "seed": 0,
  "device": "",
  "hostname": "",
  "python_version": "",
  "framework_versions": {},
  "baseline_run": "",
  "notes": ""
}

metrics.csv

Store machine-readable metrics with stable column names.

Recommended columns include:

step
wall_time
episode_return
episode_length
success_rate
position_error
orientation_error
axis_rotation
constraint_violation
actor_loss
critic_loss
entropy
approx_kl
clip_fraction
explained_variance
learning_rate

Only include metrics relevant to the algorithm and task.

summary.md

Each run must include:

* experiment question;
* change from baseline;
* final metrics;
* best metrics;
* artifact links;
* important observations;
* failure modes;
* conclusion;
* recommended next step.

checkpoints/

Save:

* periodic checkpoints;
* best checkpoint based on a clearly defined evaluation metric;
* final checkpoint;
* optimizer state when useful;
* observation or reward normalization state.

A checkpoint is not considered reproducible if required normalization statistics are missing.

⸻

8. Visualization Requirements

Training results should be made inspectable through plots, images, and videos.

8.1 Required plots

For meaningful training runs, generate when applicable:

* episodic return versus environment steps;
* evaluation success rate versus environment steps;
* task-specific error versus environment steps;
* reward component curves;
* episode length;
* policy and value losses;
* entropy;
* KL divergence or clipping fraction;
* explained variance;
* evaluation metrics with confidence intervals across seeds;
* baseline-versus-current comparison.

Plot both raw values and smoothed trends when useful. Clearly label smoothing parameters.

Axes must include metric names and units.

Plot filenames should be descriptive:

plots/train_return.png
plots/eval_success_rate.png
plots/endpoint_error_cm.png
plots/axis_rotation_deg.png
plots/reward_components.png
plots/baseline_comparison.png

8.2 Required videos

For embodied or control tasks, save evaluation videos at meaningful stages.

At minimum, capture:

* an early-training policy;
* the latest policy;
* the best-performing checkpoint;
* at least one representative success;
* at least one representative failure.

Recommended filenames:

videos/step_000000.mp4
videos/step_100000.mp4
videos/best_success_seed0.mp4
videos/common_failure_seed0.mp4

Videos should include an overlay or accompanying metadata containing:

* run ID;
* checkpoint step;
* evaluation seed;
* episode return;
* task success;
* task-specific errors;
* relevant constraint violations.

Do not select only visually impressive episodes. Preserve representative failure cases.

8.3 Task-specific visualizations

For in-hand manipulation, reorientation, or screw-driving-style tasks, generate when applicable:

* object position trajectory;
* endpoint trajectory relative to the desired fixed point;
* object-axis rotation over time;
* orientation error over time;
* fingertip contact state;
* contact force or tactile feature heatmap;
* slip events;
* action magnitude;
* joint trajectories;
* reward-component timeline;
* termination cause;
* initial and final object pose;
* success and failure episode comparison.

For a rod-axis rotation task, prefer a visualization containing:

1. endpoint position error over time;
2. unwrapped rotation angle around the rod’s own axis;
3. contact state over time;
4. episode video;
5. final numerical summary.

8.4 Visual artifact integrity

After creating a plot or video, verify that:

* the file exists;
* the file is non-empty;
* the media can be decoded;
* labels are readable;
* the content corresponds to the intended run;
* paths are recorded in the run summary and experiment log.

⸻

9. Metrics and Evaluation Protocol

Maintain docs/METRICS.md to define every important metric mathematically and operationally.

For each metric, document:

* name;
* formula;
* units;
* aggregation method;
* evaluation frequency;
* success threshold;
* edge cases;
* implementation location.

Do not silently redefine metrics between experiments.

For example:

## Endpoint Position Error
Definition:
The Euclidean distance between the designated rod endpoint and its target point.
Unit:
Meters in raw logs and centimeters in plots.
Aggregation:
Mean over timesteps, final-step value, and maximum value per episode.
## Axis Rotation Progress
Definition:
The unwrapped angular displacement around the object's local longitudinal axis relative to the initial pose.
Unit:
Radians in code and degrees in reports.
Important:
The value must be unwrapped to avoid discontinuity at ±π.

Evaluation must be separated from training rollouts.

Unless otherwise specified:

* use deterministic policy evaluation when supported;
* use fixed evaluation seeds for run-to-run comparison;
* also test additional unseen seeds;
* report the number of evaluation episodes;
* preserve per-episode results, not only averages;
* report mean, standard deviation, median, minimum, and maximum where useful;
* use multiple training seeds before making strong claims.

⸻

10. Reward Debugging

When modifying rewards, the agent must inspect individual reward components.

Do not rely only on total return.

For each reward component, log:

* raw component value;
* weighted value;
* cumulative contribution;
* minimum and maximum;
* mean per episode;
* relative share of total reward.

Check for:

* one component dominating all others;
* reward hacking;
* discontinuities;
* incorrect signs;
* unit mismatch;
* excessive scaling;
* sparse signals;
* impossible success bonuses;
* termination incentives;
* policies exploiting simulator artifacts.

Before accepting a reward change, produce at least one comparison of reward components between the baseline and modified run.

⸻

11. Training Failure Taxonomy

Classify failures instead of describing all failures as “training did not work.”

Use one or more of the following labels:

* environment bug;
* incorrect observation;
* invalid action scaling;
* reward specification failure;
* reward hacking;
* exploration failure;
* optimization instability;
* value-function failure;
* policy collapse;
* premature convergence;
* insufficient training;
* curriculum mismatch;
* contact-model issue;
* simulator instability;
* numerical instability;
* evaluation bug;
* checkpoint-loading bug;
* domain randomization too strong;
* domain randomization too weak;
* seed sensitivity;
* success detector error;
* visualization mismatch;
* unknown.

Record the selected failure type in the debugging or experiment log.

⸻

12. Code Modification Rules

Before modifying code:

1. inspect the current project state and recent logs;
2. identify the baseline behavior;
3. explain the intended change;
4. identify files likely to be affected;
5. define how the change will be tested.

After modifying code:

1. run formatting and static checks where available;
2. run relevant unit tests;
3. run an environment smoke test;
4. run a short training smoke test;
5. inspect logs for NaN, Inf, shape errors, or abnormal values;
6. generate at least one evaluation artifact if behavior changed;
7. update project records.

Do not perform large unrelated refactors during experimental debugging.

Do not change default hyperparameters without recording the change.

Do not overwrite previous experiment outputs.

Do not reuse a run directory.

⸻

13. Testing Requirements

Add tests for bugs that can be tested deterministically.

Important test targets include:

* environment reset;
* observation shape and bounds;
* action clipping and scaling;
* reward component signs and values;
* rotation-angle unwrapping;
* quaternion normalization;
* coordinate-frame conversions;
* termination conditions;
* success detection;
* checkpoint save and restore;
* evaluation determinism;
* metric calculations;
* artifact-generation scripts.

A resolved deterministic bug should normally have a regression test.

⸻

14. Experiment Comparison

When comparing runs, use the same:

* evaluation seeds;
* number of episodes;
* environment version;
* task distribution;
* success definition;
* metric implementation;
* checkpoint-selection rule.

Generate a comparison report under:

reports/comparisons/<baseline>_vs_<candidate>.md

The report should include:

# Run Comparison
## Compared Runs
## Experimental Difference
## Evaluation Protocol
## Metric Comparison
## Training Curves
## Representative Videos
## Success Cases
## Failure Cases
## Statistical Caveats
## Conclusion
## Recommendation

Never compare two runs only by their maximum training return.

⸻

15. Communication Style

When reporting progress, use the following structure:

## Current Finding
What has been observed so far.
## Evidence
Metrics, logs, plots, videos, or code references.
## Interpretation
What the evidence probably means.
## Change Made
What was changed and why.
## Verification
What tests or experiments were run.
## Remaining Uncertainty
What has not yet been proven.
## Next Step
The next smallest useful action.

Be explicit about uncertainty.

Use phrases such as:

* “The evidence suggests…”
* “This has not yet been confirmed…”
* “The run supports the hypothesis that…”
* “This result is inconclusive because…”
* “A multi-seed evaluation is still required…”

Do not say “the model is fixed” or “the method works” without sufficient verification.

⸻

16. Session Completion Checklist

Before finishing a meaningful working session, verify:

* Code changes are saved.
* Relevant tests were run.
* The active run has a unique ID.
* The resolved configuration was saved.
* Metrics were saved in machine-readable form.
* Important plots were generated.
* At least one relevant evaluation video or image was saved.
* New bugs were added to DEBUG_LOG.md.
* Experiments were added to EXPERIMENT_LOG.md.
* Reusable findings were added to FINDINGS.md.
* PROJECT_STATE.md reflects the current state.
* The best checkpoint path is recorded.
* The next experiment is explicitly stated.
* Failed or interrupted runs are clearly labeled.
* No previous experiment artifacts were overwritten.

⸻

17. Findings Database

Maintain docs/FINDINGS.md for conclusions that are likely to remain useful.

Use this format:

## FIND-YYYYMMDD-NNN: Finding title
- Confidence: low | medium | high
- Supporting runs:
- Related debug issues:
- Applies to:
- Does not apply to:
### Finding
State the reusable conclusion.
### Evidence
Summarize the evidence.
### Implication
Explain how this should affect future experiments.
### Caveats
State limitations and unresolved questions.

Examples of appropriate findings:

* a particular reward term causes the policy to sacrifice endpoint stability;
* unwrapped axis rotation is required for reliable progress measurement;
* action scale above a threshold causes contact instability;
* a certain observation is necessary for contact recovery;
* a curriculum stage is too difficult when introduced from initialization;
* evaluation success is highly seed-sensitive.

Do not copy every experiment result into FINDINGS.md. Only store durable conclusions.

⸻

18. Priority Order

When deciding what to do next, use this priority order:

1. Fix correctness bugs.
2. Fix reproducibility problems.
3. Validate metrics and success detection.
4. Establish a reliable baseline.
5. Diagnose the dominant failure mode.
6. Run the smallest discriminating experiment.
7. Improve training stability.
8. Improve task performance.
9. Run multi-seed validation.
10. Improve code structure or presentation.

Correctness and measurement validity take priority over higher reward.

⸻

19. Agent Start-of-Task Procedure

At the start of each task:

1. Read the persistent project records.
2. Inspect the current Git status.
3. Identify the current baseline run.
4. Check whether the proposed idea has already been tested.
5. State the concrete objective for this iteration.
6. State the evidence required to consider it successful.
7. Prefer the smallest experiment capable of answering the question.

If records are missing, create them before conducting extensive new experiments.

⸻

20. Agent End-of-Task Report

At the end of each task, report:

## Completed
## Files Changed
## Tests and Experiments Run
## Run IDs
## Numerical Results
## Generated Artifacts
## Debugging Knowledge Added
## Current Conclusion
## Remaining Problems
## Recommended Next Step

Include actual artifact paths and metric values.

Do not provide only a narrative summary when files, runs, or metrics were produced.

⸻

21. Prohibited Behaviors

The agent must not:

* repeatedly tune hyperparameters without a documented hypothesis;
* overwrite old checkpoints, metrics, plots, or videos;
* report only successful episodes;
* hide failed runs;
* compare runs under different evaluation protocols without disclosure;
* conclude from a single random seed when seed sensitivity is plausible;
* change metric definitions silently;
* use total reward as the only measure of task success;
* call a run successful solely because a video looks plausible;
* launch a long run before completing a smoke test;
* mark a bug resolved without verification;
* claim causality when multiple important variables changed;
* forget to update the persistent project records;
* generate visualizations without recording which run produced them;
* leave experimental changes undocumented in configuration files.

⸻

22. Default Decision Rule

When uncertain, do not immediately launch another long training run.

First ask:

1. What exact failure are we trying to explain?
2. What are the competing hypotheses?
3. What is the cheapest experiment that distinguishes them?
4. What numerical and visual evidence will resolve the question?
5. How will the result be recorded for future sessions?

The repository should become more understandable after every experiment, regardless of whether the model improves.