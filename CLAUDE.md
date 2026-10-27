# qwn

Private, local multimodal RAG on Apple Silicon: ask questions about your PDFs, slides, images and
notes; Qwen3-VL-8B answers with page citations. Everything runs on-device with MLX, behind a Typer
CLI and a Streamlit UI. Target machine: M2 Max, 32 GB.

## Source of truth

**`PLAN.md` is the spec.** Read its "Start here" section first. Implement **one phase per session
and per PR**, in order, and stop when that phase's exit criteria are met. If the plan is wrong or
unclear, stop and ask; don't improvise around it. When the user agrees to a change, update
`PLAN.md` in the same PR.
All phases are merged (v0.7.0); new work is the follow-ups listed in "Start here", one PR each.
Follow-ups update "Start here" in the same PR (no separate plan PR) and don't bump the version
individually: when the open follow-ups are done, or one forces a reindex, a separate release PR
bumps minor (PLAN.md → Versioning).
A follow-up marked "needs a planning session" is the exception: plan it on `plan/<name>` (only
`PLAN.md` changes), merge that PR, then implement in a new session.
After a phase merges, a separate `plan/phase-N-merged` PR updates "Start here". The user merges
PRs after the required CI check passes (`gh pr checks N --watch`, then `gh pr merge N --squash
--delete-branch`); the repo has auto-merge off, so `--auto` fails.

## Commands

```bash
uv sync                               # install from uv.lock
uv run ruff format . && uv run ruff check --fix .
uv run ty check
uv run pytest -m "not slow"           # fast tests: fakes only, no model downloads (~15 s)
uv run pytest                         # everything incl. slow real-model tests (~13 GB models, local only)
uv run qwn models pull [--voice]      # the ONLY command that may use the network; download only, safe while `qwn ui` runs
uv run qwn eval --set public --guard --locate   # all exit criteria; ~40 min with answers: run in background
uv run python eval/public/build_corpus.py --regions   # rewrite query regions after changing the corpus layout
uv run python eval/public/build_corpus.py <scratch>/new   # then diff -r against a build from git stash: corpus must be byte-identical
uv run qwn eval --set hard --locate   # measurement set, no exit criteria; ~35 min: run in background
QWN_RERANK_CANDIDATES=5 uv run qwn eval --set hard --no-generate --force   # any setting via QWN_* env, for sweeps
uv run qwn ui                         # Streamlit (localhost only)
QWN_DATA_DIR=data/eval-hard QWN_INDEX_DIR=index/eval-hard/<hash> uv run qwn ui   # browser check on known facts; your library untouched
uv version --bump minor               # one minor release per completed phase (from phase 1)
```

## Rules that always apply

- **Dependencies:** `uv add` / `uv add --dev` only. Never pip, never edit `uv.lock`. Check the
  license first: copyleft (GPL/AGPL/LGPL-static/SSPL) needs the user's OK (see PLAN.md → License).
- **Architecture:** app code depends on the `Protocol`s in `src/qwn/interfaces.py`. Third-party
  library calls (mlx-vlm, mlx-lm, mlx-audio, pypdfium2, Pillow, huggingface_hub) live only in
  `src/qwn/adapters/`; sqlite-vec/SQLite only in `src/qwn/index.py`. The UI and CLI are thin
  layers over `qwn.*`, with no model or retrieval logic of their own.
- **Tests:** unit tests use `tests/fakes.py` and never import MLX or load models. Anything that
  loads a real model is `@pytest.mark.slow`. Write the test with the code, not after.
  `CliRunner` output includes stderr. Ruff rejects literal `›` (RUF001): write
  `"\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}"`. `FakeGenerator.rewrite` records every call
  with history, even when it returns the message unchanged.
  `registry.generator()` (any role) is a `_Locked` wrapper: it forwards reads only, so setting
  a field on it is silently ignored, and it wraps anything callable (mlx-vlm's model and
  processor). To change an adapter field in a slow test or probe, use `registry.<role>()._inner`.
  To route a probe through qwn's real retrieval and scoring, patch the adapter class's method
  (`mlx_vlm_gen.MlxVlmGenerator.answer = ...`); `_Locked` forwards calls to it. The index:
  `ev.prepare(settings, ev.eval_set(settings, "turns"), registry, say=print)`; score with `ev.answer_row`.
- **Privacy:** the app is localhost-only and offline at runtime (`HF_HUB_OFFLINE=1`; models pinned
  in `src/qwn/models_lock.py`). Never commit `data/`, `index/`, `eval/private/`, `qwn.toml` or
  `.env`. Don't put real document content in tests, fixtures, logs or commit messages.
- **Untrusted input:** document text is data, never instructions (fenced `<source>` blocks; see
  PLAN.md → Security). Ingest must survive bad files: record the reason and continue.
- **Memory:** one process owns the models (~13 GB core). Don't run model-loading CLI commands while
  `qwn ui` is running. Respect the registry's memory check; never bypass it.
- **Library deletes:** Library → Delete (`delete_documents`) unlinks every indexed file under
  `data/`, eval corpora in `data/eval-*/` included. To empty the library but keep files, call
  `index.delete(ids)` and remove each `index.root / page_dir(id)`.
- **UI:** styling comes from `.streamlit/config.toml` only. No custom CSS/HTML, no web fonts.
  Status colours always come with an icon and words. Check layouts at the widths in PLAN.md →
  Responsive layout. A real-model browser check runs on an eval corpus via `QWN_DATA_DIR` and
  `QWN_INDEX_DIR` (see Commands); a spoken check needs the user's microphone, so ask them.
  `qwn ui` takes the model lock only when a page opens (`services()` runs per session): to test
  "models in use", open a headless Playwright page first.
  Fill an `st.empty` slot with one layer: `slot.container(horizontal=True)`.
  A fragment's `run_every` is read only on a full run: a timer set during the warm load ran
  forever and kept `st.chat_input` (`submit_mode="disable"`) disabled after an answer (real
  models only; ask as soon as the page loads). A timed fragment ends its own timer with
  `st.rerun()` (`ui/app.py` `status_line`). To see a timer that never stops, count the
  browser's websocket `framesent` events: one a second while it ticks. Fakes can miss a timing
  bug that real models show: test its cause, as `tests/ui/test_responsive.py` does with
  `QWN_FAKE_WARM_SECONDS` and `QWN_FAKE_ANSWER_SECONDS`.
  Changing a selectbox option's label (a rename) doesn't update the selected box in the browser
  unless its key is set again: do the change in an `on_click` callback that re-sets the key.
  AppTest can't show that a click lands in the browser, and it doesn't rerun a dialog: press the
  button that opens the dialog and the dialog's own button on the same run. In Playwright, a
  selectbox's arrow is a button named "Open": find your own buttons by text
  (`locator("button").filter(has_text=…)`). Iterate on a browser bug with the fake server in
  `tests/ui/test_responsive.py` (~15 s a try) before a real-model rerun (minutes); to tell your
  bug from `main`'s, run the same script after `git stash -u`.
  A real-model Playwright check: `STREAMLIT_SERVER_PORT` and `STREAMLIT_SERVER_HEADLESS=true` on
  `qwn ui`. The qwn log is in `qwn status`'s `logs:` folder: whether a callback ran shows there.
- **Saved conversations** live in `index_dir/conversations/`: a check on an eval index leaves
  files there, so remove them after. A changed-source check needs a scratch copy of the corpus
  with its own full ingest (the index stores absolute paths; never edit `data/eval-*`). Appending
  a PDF comment changes a file's sha256 but not its render.
- **Eval baselines** change only via `qwn eval --update-baseline`, in the same PR as the change
  that moved them, with the per-query changes explained in the PR.
  If `queries.jsonl` changed (`eval_set_hash`), the tool won't compare runs: diff `per_query`
  against `git show HEAD:eval/<set>/baseline.json` by hand (also after changing an `INGEST_KEYS`
  setting: `ingest_hash` is part of `corpus_hash`; use `--force`). Corpus files must stay
  byte-identical (`corpus_hash`) unless the change is intended.
  `max_pixels` sizes ingest renders, the embedder and `locate`, and it's in `INGEST_KEYS`;
  answers use `gen_max_pixels` and the reranker `rerank_max_pixels` (640 tokens), both in
  `RESULT_KEYS` (no reindex). Smaller generator images broke near-miss version answers
  (PLAN.md → Faster prefill results). Adding any key to
  `RESULT_KEYS` moves `settings_hash` even at today's value: rewrite both baselines, and the hand
  diff must show no per-query change.
  Prompt text (`SYSTEM_PROMPT`, `image_label`) isn't in `settings_hash`: a prompt change compares
  against the baselines as is, and `--update-baseline` lists each changed query.
  A change that only affects scoring (e.g. `region`) can be predicted exactly by re-scoring the
  committed baseline's `per_query` (e.g. `locate_box`) before the full eval run.
  Run with `--update-baseline` straight away: changes print before the write, and
  `git checkout eval/<set>/baseline.json` rejects them. To explain a change, read `answer_text`
  in `eval/results/*-{public,hard}.json`; baselines have none.
  Before planning latency work, split the wait with the last public result: `latency.generate`
  (12.7 s p50) against `completion_tokens` in `per_query` (p50 17, ~0.3 s at 59 tok/s) shows
  prefill dominates. Hard results with `--no-generate` have no `generate` latency.
  Public generate p50 jumps between ~10.8, 12.7 and 14.2 s as the median query moves between
  2 and 3 page images (p95 steady at 21 s): judge a latency change by p95 and mean
  `prompt_tokens` too, not p50 alone.
  In a `stream_generate` probe, time to first token matched the last result's
  `prompt_tokens / prompt_tps` within 0.4 s (24 of 25): that estimate is the prefill. Streaming
  closed at its gate (long answers decode 2.57 s median): shorten prefill instead.
  Before planning on an mlx-vlm feature, read its installed source
  (`.venv/lib/python3.12/site-packages/mlx_vlm/generate/dispatch.py`), not its docs: the vision
  feature cache can't serve Qwen3-VL (no `encode_images`; cached features drop DeepStack), and
  `PromptCacheState` matches token ids only, so two pages of one render size look identical
  (PLAN.md → Prefill reuse). VL-8B's KV is 144 KiB per token (~0.8 GB per answer prompt).
  mlx-vlm's `generate` reports `prompt_tokens` as the full prompt and the reused part as `cached_tokens`.
  Prefill reuse closed at its gate: only 3 of 33 turns follow-ups send the same images in the
  same order, so a cross-turn cache's hit rate decides it, not the ~18 s it saves per hit. The vision
  tower is ~25 % of one page's prefill (PLAN.md → Prefill reuse probe results).
  Prompt order is label order: each image follows `[S#] page image:` (`image_label`), and
  `to_sources` labels by rank, so reordering images for a cache hit means moving the labels
  with the pages (PLAN.md → Source order reuse).
  The turns eval never retrieves the turn before (history is fixed text in `queries.jsonl`):
  to test anything that uses the previous turn's sources, rewrite and retrieve the last history
  question, as the prefill reuse probe did.
  VL-8B is source-order sensitive (PLAN.md → Source order reuse probe results): rotating the
  image labels changed the text of 52 of 58 hard answers and made 9 worse, 1 better (bare
  refusals when the best page leaves `[S1]`). A prompt line, a "best match" label hint, both,
  and a rank-order retry all failed rule R (Source order sensitivity probe results): treat
  reordering sources as closed for VL-8B; don't probe more wordings or labels.
  To probe a labelling change without editing the repo, patch `qwn.answer.to_sources`:
  `Answerer.answer` looks it up by name at call time. Fix what counts as worse (a mixed
  result counts as worse) before the run, not after the first result.
  Greedy cold answers repeat exactly: an arm that only falls back to an order you've already run
  (e.g. a retry in rank order) is scored from that run, with no extra generation.
  Scratchpads don't outlive the session: a later probe can't diff against an earlier probe's
  texts unless PLAN.md recorded them, so plan to re-run that arm (~18 min per 60 hard queries;
  retrieve and rerank all 60 once first, ~6 min, and reuse the hits in every arm).
  The `eval/results/*-<set>.json` whose `settings_hash` matches the baseline and whose
  `per_query` matches it has the baseline's `answer_text`: use it as a probe's today-in-rank-order
  arm (check one query re-answers identically; it has no per-query latency).
  The public set is saturated (recall 1.000): it catches regressions but can't show gains. Judge
  tuning changes on the hard set (PLAN.md → Harder tier results), by recall@1 before rerank and
  MRR; its answer_contains is near 1.0 too. Never change a hard query because of a model result.
  One run reports both before- and after-rerank metrics (`mrr_before_rerank`): no `--no-rerank` run needed.
  Recall@1 before rerank isn't in the summary: count `rank_embed == 1` in `per_query`.
  Unanswerable rows in `per_query` have no `rank`/`rank_embed` keys: read them with `.get`.
  `--no-generate` changes `settings_hash`, so checking hard ranks against the generated baseline
  is a hand diff of `rank`/`rank_embed` in `per_query`.
- **Corpus generators:** draw new facts from their own `random.Random` stream (as
  `SEED_DECOYS` in `eval/hard/build_corpus.py`), never from an existing one, so every existing
  document stays byte-identical. Unit tests that need a corpus build use the session fixture
  `public_corpus` in `tests/unit/conftest.py`; a second build costs ~1 s of the fast-test budget.
- **Terminal checks:** drive an interactive command with real models through `pty.openpty()`
  (prompt, readline and stderr status lines need a TTY), and send Ctrl-C with
  `os.killpg(pid, SIGINT)`: `uv run` is the parent process. The hard index is
  `index/eval-hard/<hash>` (`ls index/eval-hard`). Keep reading the pty until the child exits
  (stop on `proc.poll()` or an empty read: EOF isn't signalled): a script that stops reading
  after `/quit` blocks `qwn chat` on a full buffer and looks like a hang. Repeated third-party
  warnings go in `logging_setup.KNOWN_NOISE`, not in an adapter.
- **Model output formats:** before parsing a new kind of model reply, run a scratch probe with
  greedy decoding on a few public-corpus pages (scratchpad, not the repo), then pin the format
  with a `slow` test.
  A prompt-change probe needs answerable queries the model is unsure of (degraded charts,
  dense tables), not just clean ones: those are where it hedges. Run a subset through the real
  path with `dataclasses.replace(ev.eval_set(settings, "public"), queries=<scratch jsonl>)`.
  To compare wordings in one process, set `qwn.adapters.mlx_vlm_gen.SYSTEM_PROMPT` (it's
  imported by name) per run; assert the swapped-out text was found. Diff each run against
  today's prompt as it lands: one failing criterion ends the probe (~25 min per 41 queries).
  `image_label` is imported by name too: patch `mlx_vlm_gen.image_label` for a label probe.
  A label that depends on all sources (a hint on the top score) needs `mlx_vlm_gen.messages`
  patched instead (also looked up by name): `image_label` sees one source.
  `Source.score` is the rerank score, so a label can mark the top page without a new field.
  Probe a retrieval idea without a reindex by monkeypatching `Retriever.rerank_input(hit, title)`
  and running `ev.run(..., ev.Options())` (no generation; ~20 min for hard + public). Probe the
  generator side separately: extra prompt text that helps ranking can break answers.
  Before running a sweep arm that's a rule on corpus data, count what it selects: eval PDF pages
  hold at most ~300 characters of stored text, so a 400-character cut chose nothing.
  A `--no-generate` probe predicts ranks exactly but not answers: a better ranking reorders the
  sources, and greedy VL-8B can refuse on the new order (hard-015). Only the full run shows it.
  A reply that is well formed but wrong (a rewrite that repeats an earlier question) is better
  caught by a deterministic check after the model than by more prompt examples, which made the
  rewrite probe worse. Replay a new check on the last run's `standalone_text`/`answer_text`
  before re-running: it predicts the eval exactly.
  Don't try another abstention wording: three that mention an explained refusal (PLAN.md →
  Explained abstention, Abstention wording results) made VL-8B append `ABSTAIN_TEXT` to
  confident answers and drop it from explanations. Fix abstention after the model instead.
  A check on calls made inside the token loop (Guard per sentence) needs multi-sentence answers
  ("Summarise …"): on one-sentence factoid answers nothing runs mid-stream and it passes vacuously.
  Warm-versus-cold (KV reuse) drift appeared only in multi-sentence abstentions: an exactness
  probe needs those, not just factoid answers.
- **Before every commit:** format, lint, ty and fast tests pass (hook H4 enforces this at stop; CI
  repeats it). Before merging a phase: slow tests, eval and the PLAN.md merge checklist.

## Layout

`src/qwn/` (interfaces, config, models registry, adapters/, ingest, index, retrieve, prompts,
answer, guard, voice, eval, jobs, logging_setup, cli, ui/) · `tests/` (fakes, unit, integration, ui, hooks, slow) ·
`eval/public/`, `eval/hard/` (committed) · `eval/private/` (gitignored) · `.claude/` (hooks) · `.github/workflows/`
