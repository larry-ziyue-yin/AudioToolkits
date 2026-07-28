# NCMMSC 2026 evaluation handoff for 14.18

Last updated: 2026-07-27 00:47 CST
Target host: `smiip1418` (`yinzy@10.200.14.18`)

## Mission

Take over and finish all NCMMSC 2026 six-metric evaluations on 14.18. Inference is
already complete, and 14.17 has been released for new training. Do not schedule
evaluation work on 14.17.

The user's requirements are:

- One experiment configuration corresponds to one tmux session.
- All conditions belonging to that configuration stay in the same tmux.
- Every active evaluation uses all eight GPUs.
- Rotate `CUDA_VISIBLE_DEVICES` so different evaluations do not all place their
  logical `cuda:0` process on physical GPU 0.
- Run at most three full-eight-GPU evaluations concurrently on the eight RTX
  3090s. Four concurrent evaluations caused a real OOM.
- Use exactly six metrics: `secs,cer,wer,utmos,wvmos,nisqa`.
- Preserve the original `evaluation` directory and resume from intermediates.
  Do not create `evaluation_attempt_*` directories and do not delete an
  incomplete evaluation directory.
- A completed experiment tmux must exit automatically; do not retain dead panes.
- D6-D was the highest-priority result and is already complete.

## Current state

### Inference and 14.17

- Main inference: `103/103`, failed `0`.
- Supplemental inference: `24/24`, failed `0`.
- `ncmmsc2026_all`, `ncmmsc_supplemental_1417`, and the handoff monitor have
  exited/been cleaned up.
- All eight GPUs on 14.17 were free after handoff.
- Do not restart the old `ncmmsc2026_all` controller. Its built-in evaluation
  phase is sequential, moves incomplete evaluations to attempt directories, and
  does not use the new claim/slot scheduler.

### Completed evaluation groups on 14.18

These six experiment configurations are complete and validated, seven
conditions each (`42/42` conditions):

1. `noisy_aligner_debug6_phase1_c_best_safe`
2. `noisy_aligner_debug6_phase1_d_best_safe`
3. `noisy_aligner_debug6_phase2_r0_short_best_safe`
4. `noisy_aligner_debug7_c1_d4nt_best_safe`
5. `noisy_aligner_debug7_c1_d5nt_best_safe`
6. `noisy_aligner_debug7_c1_d6d_best_safe`

Their experiment-level markers are under:

```text
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/<experiment>.success
```

### Newly launched evaluation groups

There are 13 live tmux sessions:

- Nine remaining main-registry configurations:
  `ncmmsc_eval_1418_main_00` through `ncmmsc_eval_1418_main_08`.
- Four supplemental configurations:
  `ncmmsc_eval_1418_supp_00` through `ncmmsc_eval_1418_supp_03`.

The exact session-to-experiment mappings and rotated GPU orders are recorded in:

```text
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/registry_main_sessions.tsv
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/registry_supplemental_sessions.tsv
```

At the timestamp above, three evaluations held the three slots:

| Experiment/condition | `CUDA_VISIBLE_DEVICES` |
|---|---|
| Published baseline, first pending condition | `0,1,2,3,4,5,6,7` |
| `noisy_aligner_debug6_phase1_a_best_safe/snr_25` | `7,0,1,2,3,4,5,6` |
| `noisy_aligner_debug5_best_safe/snr_25` | `4,5,6,7,0,1,2,3` |

The other ten tmux sessions are waiting for a slot. GPU memory after model load
was about 20.9 GiB on the three rotated primary GPUs and 15.7 GiB on the other
GPUs. There were no failed markers or OOMs.

Overall completion target:

- 19 complete experiment configurations: 15 main + 4 supplemental.
- 127 successful condition evaluations: 103 main + 24 supplemental.
- Current baseline at handoff: 6/19 experiment groups and 42/127 conditions.
- Failed markers at handoff: 0.

## Important paths

### Registries

```text
/home/yinzy/whispervc/egs/conf/ncmmsc2026_experiments.json
/home/yinzy/whispervc/egs/conf/ncmmsc2026_supplemental_inference.json
```

### Scheduler and worker

```text
/home/yinzy/whispervc/egs/scripts/inference/cfm_mel_based/launch_ncmmsc2026_eval_registry_1418.sh
/home/yinzy/whispervc/egs/scripts/inference/cfm_mel_based/run_ncmmsc2026_eval_experiment_8gpu.sh
```

The launcher is idempotent:

```bash
bash /home/yinzy/whispervc/egs/scripts/inference/cfm_mel_based/launch_ncmmsc2026_eval_registry_1418.sh main
bash /home/yinzy/whispervc/egs/scripts/inference/cfm_mel_based/launch_ncmmsc2026_eval_registry_1418.sh supplemental
```

Re-running it skips experiment-level success markers, preserves live sessions,
and starts only missing sessions.

### Evaluation configuration and validation

```text
/home/yinzy/AudioToolkits/w2n_eval_metrics_offline.yaml
/home/yinzy/whispervc/whisper_wavlm/utils/evaluation_validation.py
```

Important evaluation settings:

```yaml
output:
  save_intermediate: true
  overwrite: true

metrics:
  - secs
  - cer
  - wer
  - utmos
  - wvmos
  - nisqa
```

The validator requires base columns
`utt_id,gen_path,gt_path,src_path,ref_text` and the six metrics above. WavLM is
validated only if present.

### Runtime state

```text
Run root:
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel

Per-experiment logs:
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/logs/<experiment>

Condition success/failure markers:
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/status

Three host-wide evaluation slots:
/home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/slots/slot_{0,1,2}

Cross-session condition claims:
/home/yinzy/whispervc/egs/output/ncmmsc2026/_eval_claims/<task_id>
```

## How the scheduler works

Each tmux owns one complete experiment configuration. Its worker iterates all
conditions in that experiment. A condition is handled as follows:

1. Validate existing results; valid results are marked successful and skipped.
2. Verify that inference and metadata are complete.
3. Acquire the exact task claim.
4. Acquire one of three evaluation slots.
5. Evaluate in the existing `evaluation` directory using all eight visible GPUs.
6. Validate the six-metric results.
7. Write `<task_id>.evaluation_success`, release the claim and slot, and continue.
8. After every condition is valid, write `<experiment>.success` and exit the tmux.

All sessions share the same slot and claim roots. This prevents duplicate
condition evaluation and limits memory usage even though 13 tmux sessions exist.

## Monitoring commands

Run these directly on 14.18:

```bash
tmux list-sessions -F '#{session_name}|dead=#{pane_dead}|exit=#{pane_dead_status}' | sort

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

ps -eo pid,etimes,args | awk '/[a]udiotoolkits.evaluation.run/ {print}'

find /home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/status \
  -maxdepth 1 -type f -name '*.evaluation_success' -printf . | wc -c

find /home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel/status \
  -maxdepth 1 -type f -name '*.failed' -printf '%f\n' | sort

find /home/yinzy/whispervc/egs/output/ncmmsc2026/_remote_eval_1418_parallel \
  -maxdepth 1 -type f -name '*.success' -printf '%f\n' | sort
```

The last command also lists control markers such as
`registry_main_sessions_launched.success`,
`registry_supplemental_sessions_launched.success`, and
`remaining_sessions_resumed.success`. Do not count those as experiment
completion markers.

To inspect a specific session:

```bash
tmux capture-pane -p -t <session>:eval -S -50
```

Most metric output is redirected to the per-experiment log directory, so a quiet
pane is not evidence that evaluation is stuck. Check the Python process, log
mtime, intermediate cache growth, GPU activity, and status markers together.

## Failure and resume procedure

On an abnormal exit:

1. Inspect the exact condition log and `.failed` marker.
2. Confirm that no process still owns that condition.
3. Confirm that a slot or claim directory is stale before removing that exact
   directory. Never broadly delete the slot or claim roots.
4. Do not delete or rename the condition's `evaluation` directory.
5. Relaunch the missing experiment by re-running the appropriate idempotent
   registry launcher.

ASR progress is cached in:

```text
<condition>/evaluation/intermediate/asr/gen_text.txt
```

Restarting in the same directory skips cached ASR utterances. UTMOS, WVMOS, and
NISQA do not necessarily checkpoint every row, so those stages may recalculate.
Final `results.csv` and `summary.csv` are written at finalization.

Do not create or retain `evaluation_attempt_*` directories for new failures.

## Final acceptance checks

Do not declare completion based only on the absence of tmux sessions. Require:

1. All 19 experiment-specific `.success` markers.
2. Exactly 127 `.evaluation_success` markers.
3. Zero current `.failed` markers.
4. Registry validation succeeds for all tasks in both registries.
5. No evaluation Python processes, slots, or claims remain.
6. All evaluation tmux sessions have exited.

Registry validation can be run with:

```bash
repo=/home/yinzy/whispervc
output=/home/yinzy/whispervc/egs/output/ncmmsc2026
python=/home/yinzy/miniconda3/envs/w2n-cu121-py310/bin/python
tool=$repo/whisper_wavlm/utils/ncmmsc2026_registry.py

for registry in \
  "$repo/egs/conf/ncmmsc2026_experiments.json" \
  "$repo/egs/conf/ncmmsc2026_supplemental_inference.json"
do
  while IFS= read -r task_id; do
    PYTHONPATH="$repo:${PYTHONPATH:-}" "$python" "$tool" \
      --registry "$registry" verify-evaluation \
      --task-id "$task_id" \
      --repo-root "$repo" \
      --output-root "$output"
  done < <(
    PYTHONPATH="$repo:${PYTHONPATH:-}" "$python" "$tool" \
      --registry "$registry" task-ids
  )
done
```

## D6-D reference

D6-D is complete and validated for all seven conditions. Its overall CER values
were:

| Condition | Overall CER |
|---|---:|
| clean | 16.154836 |
| snr_0 | 37.295527 |
| snr_5 | 29.209734 |
| snr_10 | 22.772139 |
| snr_15 | 19.868392 |
| snr_20 | 18.128628 |
| snr_25 | 16.992582 |

The old D6-D tmux once ended with exit code 2 because its live shell script was
overwritten near the final `done`. This happened only after all seven conditions
had succeeded. Results were independently revalidated and the experiment marker
was written. Do not rerun D6-D.

Never overwrite an active Bash worker script in place. If a worker change is
unavoidable, stop the affected sessions first or deploy a versioned new script
and use it only for new sessions.

## Known handoff incident

The first automatic registry launch attempt parsed the scope as `main}` or
`supplemental}` because the shell parameter expansion contained literal braces.
This is fixed in the deployed launcher:

```bash
SCOPE=${1:-}
```

Both launch-control markers now exist, and all 13 expected new tmux sessions
were started successfully.

## Dirty worktrees: preserve existing changes

Do not clean or reset either repository.

Intentional/relevant WhisperVC changes include:

```text
M  whisper_wavlm/utils/evaluation_validation.py
?? egs/scripts/inference/cfm_mel_based/launch_ncmmsc2026_eval_registry_1418.sh
?? egs/scripts/inference/cfm_mel_based/run_ncmmsc2026_eval_experiment_8gpu.sh
?? egs/scripts/inference/cfm_mel_based/launch_ncmmsc2026_eval_parallel_1418.sh
?? egs/scripts/inference/cfm_mel_based/resume_ncmmsc2026_eval_after_d6d_1418.sh
```

The supplemental registry and inference controller are staged user changes.
There are also unrelated files such as `error.txt`; preserve them.

Intentional/relevant AudioToolkits state includes the modified
`w2n_eval_metrics_offline.yaml`, the preserved full-metric config, offline WVMOS
config, `.orig` files, and other user-owned untracked files. Do not delete or
reset them.
