# Contributing

## Branch model

Work happens on short-lived feature branches off `main`. Open a Merge Request when ready; CI must be green before merge.

## Local checks before opening an MR

```bash
source venv/bin/activate
ruff check .                      # lint — must pass
python -c "import app, pipeline"  # smoke import — must succeed
streamlit run app.py              # manual UI smoke test on a small batch
```

There is no automated test suite for the image pipeline; the CI smoke job verifies that the modules import cleanly under a fresh venv. UI changes need a manual visual pass.

## CI workflows

Every workflow lives in [`.github/workflows/`](../.github/workflows/):

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| [`smoke.yml`](../.github/workflows/smoke.yml) | push to `main`, PR | Builds a fresh venv on Ubuntu, installs `requirements.txt`, imports `app` and `pipeline`. Catches dependency drift. |
| [`ruff.yml`](../.github/workflows/ruff.yml) | push to `main`, PR | Runs `ruff check` against the repo. Config in [`ruff.toml`](../ruff.toml). |
| [`codeql.yml`](../.github/workflows/codeql.yml) | push to `main`, PR | GitHub CodeQL static analysis for Python. |
| [`gitleaks.yml`](../.github/workflows/gitleaks.yml) | push to `main`, PR | Scans for accidentally-committed secrets. |
| [`release.yml`](../.github/workflows/release.yml) | push of a tag | Reads `VERSION`, creates the GitHub release, attaches artifacts. |
| [`dependabot-automerge.yml`](../.github/workflows/dependabot-automerge.yml) | dependabot PRs | Auto-approves and auto-merges patch/minor dependency bumps that pass CI. |

## Code style

- **Linter:** ruff — see [`ruff.toml`](../ruff.toml). All rules enforced are blocking.
- **Imports at the top of the file** (E402). Lazy loading of heavy libraries is done by importing them *inside functions in `pipeline.py`*, not by reordering import statements. See commit `ba87057`.
- **Formatting:** ruff's formatter is the source of truth. Run `ruff format .` before committing.
- **Type hints** are encouraged on public functions in `pipeline.py`; not enforced on internal helpers.
- **No emoji** in source files. Glyphs in the status console (`✓ ✕ ▸`) are intentional UI markers, not decoration.

## Commit messages

Style observed in the repo: short imperative subject, optional body for context.

```
fix: move module-level imports to top of file (ruff E402)
chore: bump version to 1.5.1 and standardize string formatting across app.py
Resume on restart, RAM-aware worker cap, no more OOM on big batches
Parallel batch processing with a thread-pool worker model
```

Conventional-commit prefixes (`fix:`, `chore:`, `feat:`) are used when helpful, but not required.

## Release flow

1. Land all PRs targeting the release.
2. Bump [`VERSION`](../VERSION) (e.g., `1.5.1` → `1.5.2`).
3. Commit: `chore: bump version to <new>`.
4. Tag: `git tag v<new> && git push --tags`.
5. [`release.yml`](../.github/workflows/release.yml) takes over: it cuts the GitHub release using `VERSION`.

For breaking changes, also update [Changelog](Changelog).

## Modifying the matting tuning

The constants in [`pipeline.py:100 – 135`](../pipeline.py) (alpha-matting thresholds, decontam blend bounds, bilateral filter parameters) encode the chapter's quality contract. Treat changes here as design decisions:

1. Run a representative batch (10 – 20 images) before and after the change.
2. Diff the `_nobg.png` outputs side by side, paying attention to hair and translucent edges.
3. Document the rationale in the commit message.

Past commits to learn from: `c3eb111`, `bf1a3a3`, `748a426` (see [Changelog](Changelog)).
