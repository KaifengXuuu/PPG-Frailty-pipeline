# Root deployment and Git publishing

Publish the **contents** of the V6 distribution, not an enclosing `v6/` folder.
The resulting checkout contains the V6 entry points, `src/`, `configs/`,
`assets/`, manifests/splits, tests, documentation, model bundles, and its own
`LICENSE`, `.gitignore`, and `.gitattributes`. Keep `.gitattributes`: it prevents
Git line-ending conversion from changing hash-bound inputs. Only the raw
databases are supplied separately:

```text
repository/
  .git/
  .gitignore
  README.md
  pyproject.toml
  pipeline.py, sweep.py, dashboard.py, analyse_report.py, ...
  src/, configs/, assets/, manifests/, splits/, tests/, ...
  PPG_Testing_05_01_2026/    # local only; ignored by Git
  physionet.org/            # local only; ignored by Git
```

Keep the original checkout until the new branch is verified and merged. These
commands do not delete or rewrite the original checkout or Git history. The
old files are removed only from the new branch's tracked snapshot. A normal
merge retains the previous commits; `.gitignore` is not a history purge.

## 1. Clone into an unused folder and create the release branch

Run these Bash commands in the same terminal. Adjust the destination path if
needed; it must not already contain a checkout. Do not run this procedure in
the original repository.

```bash
V6_SOURCE_REPO=/home/trinker/Code/github/PPG-Frailty-pipeline
V6_SOURCE_DIR="$V6_SOURCE_REPO/final_v0/final_pipeline_v6"
V6_PUBLISH_REPO=/home/trinker/Code/github/PPG-Frailty-v6-publish
V6_RELEASE_BRANCH=release/v6-standalone
V6_REMOTE=$(git -C "$V6_SOURCE_REPO" remote get-url origin)

test -f "$V6_SOURCE_DIR/pipeline.py"
test -f "$V6_SOURCE_DIR/assets/authority/manifests/frailty3_file_manifest.csv"
git clone --filter=blob:none --no-checkout --branch main "$V6_REMOTE" "$V6_PUBLISH_REPO"
cd "$V6_PUBLISH_REPO"
git switch -c "$V6_RELEASE_BRANCH"
```

`--no-checkout` deliberately leaves the old tracked files out of the working
tree. `git status` will initially show their staged deletion. The new branch
still starts from `main`; it is not an orphan branch. `--filter=blob:none`
avoids downloading old file contents until Git needs them. If the server does
not support partial clones, omit that option; `--no-checkout` still applies.

## 2. Copy the distribution, including its ignore rules

```bash
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.pytest_cache/' --exclude='*.egg-info/' \
  --exclude='/cache/' \
  --exclude='/pipeline_output/' --exclude='/report_output/' \
  --exclude='/PPG_Testing_05_01_2026' --exclude='/physionet.org' \
  "$V6_SOURCE_DIR/" "$V6_PUBLISH_REPO/"
```

The trailing slash on the source means "copy its contents". Do not add
`--delete`, and do not replace this command with a recursive deletion of the
repository. Small `assets/`, `model_config/`, and required `artifacts/` files
must be copied: they are inputs, not disposable result caches. Output folders
are created automatically on first use.

## 3. Supply the two databases locally

Copying the complete original directories currently requires approximately
17 GiB of additional space. Check disk space first:

```bash
df -h "$V6_PUBLISH_REPO"
du -sh "$V6_SOURCE_REPO/PPG_Testing_05_01_2026" "$V6_SOURCE_REPO/physionet.org"
rsync -a --info=progress2 \
  "$V6_SOURCE_REPO/PPG_Testing_05_01_2026/" "$V6_PUBLISH_REPO/PPG_Testing_05_01_2026/"
rsync -a --info=progress2 \
  "$V6_SOURCE_REPO/physionet.org/" "$V6_PUBLISH_REPO/physionet.org/"
unset PPG_FRAILTY_DATA_ROOT
git check-ignore PPG_Testing_05_01_2026 physionet.org
```

The last command must list both database names. Neither database will be
uploaded. To avoid copying, leave both databases in a persistent private data
folder and set `PPG_FRAILTY_DATA_ROOT` to their common parent instead. In that
case, do not delete that data folder and do not run the `unset` command.

## 4. Install and validate

For the reference numerical environment:

```bash
conda create -n ppg-v6 python=3.11.14
conda activate ppg-v6
python -m pip install -r requirements/requirements-finalcase.txt
python -m pip install -r requirements/requirements-artifact-legacy-ablation.txt
python -m pip install --no-deps -e .
python -m pip check

python -m pytest -q
python pipeline.py validate --preset finalcase --mode full
python sweep.py validate --plan configs/studies/finalcase.yaml
python specialized_pipeline.py validate --plan configs/studies/thesis/motion_detector.yaml
python specialized_pipeline.py validate --plan configs/studies/thesis/denoiser.yaml
python dashboard.py --host 127.0.0.1 --port 8050
```

Open `http://127.0.0.1:8050`, check the controls/previews, then press Ctrl+C
in the terminal to stop Dash. These validation commands do not run a full
experiment. To use an existing suitable environment, activate it instead of
creating `ppg-v6`, but re-run the installation checks and tests. Optional PRV
comparison backends have separate requirement files and may require isolated
environments; do not merge those incompatible requirements into the main
numerical environment.

To actually train finalcase and render its saved results:

```bash
python sweep.py run --plan configs/studies/finalcase.yaml --run-name finalcase_v6_01
python analyse_report.py run --input pipeline_output/finalcase_v6_01 --preset classification
```

Use a new run name for a new training run. The thesis experiment index is
[`configs/studies/thesis/README.md`](../configs/studies/thesis/README.md).

## 5. Review, commit, and push only the new branch

Stage the release before generating optional experiment outputs if you want a
code-only release. `pipeline_output` results are intentionally eligible for
later versioning; `report_output` and caches are ignored.

```bash
git add --all
git status --short
git diff --cached --name-status
git diff --cached --check
git ls-files -- PPG_Testing_05_01_2026 physionet.org
```

The last command must produce **no output**. Also check that no `.env`, raw
recordings, cache, private results, or unexpected symlinks have been staged.
Review bundled manifest identifiers/labels and `learned_model/golden.npz`
verification samples before publishing to a public repository; ignoring raw
CSVs alone does not anonymize the distribution. Old source-file deletions are
expected; missing V6 algorithms, authority assets, or models are not.

If optional training was already run and its results should remain local:

```bash
git restore --staged -- pipeline_output
```

This only unstages output files; it does not delete them. Run it only if that
path was staged. Then commit and push:

```bash
git commit -m "Deploy standalone English V6 pipeline at repository root"
git push -u origin "$V6_RELEASE_BRANCH"
```

## 6. Merge into main

Recommended: open a pull request on GitHub with base `main` and compare
`release/v6-standalone`. Review the deletions/additions and test result, then
merge the pull request. Afterwards synchronize the local checkout:

```bash
git fetch origin main:main
git switch main
```

Run the fetch while still on the release branch: it fast-forwards the local
`main` reference before switching. Do not check out the old local `main` first,
because it may still track database paths now occupied by ignored local copies.
The merged tree no longer tracks those directories. Switch branches only with
a clean tracked working tree; the fetch refuses to overwrite a diverged local
`main`.

Alternatively, if direct pushes to `main` are allowed, integrate any new main
commits on the release branch and then fast-forward the remote main reference:

```bash
git fetch origin
git merge --no-edit origin/main
python -m pytest -q
git push origin "$V6_RELEASE_BRANCH"
git push origin HEAD:main
```

Run this alternative while still on `release/v6-standalone`. The merge is made
there so no checkout of old tracked database files is needed. If conflicts or
tests fail, resolve them and re-test before either push. If `main` advances
again, fetch and merge again. Never use `--force` or rewrite `main` history.
Branch protection may require the pull-request route instead.
