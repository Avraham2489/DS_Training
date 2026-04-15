# DS Training

Training materials for the Data Science program.

---

## Table of Contents
- [Course Structure](#course-structure)
- [Getting Started (do this once)](#getting-started-do-this-once)
- [Submitting an Exercise](#submitting-an-exercise)
- [PR Title Format](#pr-title-format)
- [Getting New Material](#getting-new-material)
- [Exams](#exams)

---

## Course Structure

| Folder | Topic |
|---|---|
| `A - Git` | Git & GitHub basics |
| `B - Python` | Python basics, advanced Python, NumPy, Pandas |
| `C - Statistics` | Statistics |
| `D - ML` | Machine Learning — regression, supervised, unsupervised |
| `Optional` | Visualization and extras |

---

## Getting Started (do this once)

### Step 1 — Fork the repo

1. Go to [github.com/Avraham2489/DS_Training](https://github.com/Avraham2489/DS_Training)
2. Click **Fork** (top right corner)
3. Leave all settings as default and click **Create fork**

You now have your own copy at `github.com/<your-username>/DS_Training`.

### Step 2 — Clone your fork

```bash
git clone https://github.com/<your-username>/DS_Training.git
cd DS_Training
```

### Step 3 — Connect to the instructor's repo

```bash
git remote add upstream https://github.com/Avraham2489/DS_Training.git
```

Verify it looks right:

```bash
git remote -v
# origin    https://github.com/<your-username>/DS_Training.git (fetch)
# origin    https://github.com/<your-username>/DS_Training.git (push)
# upstream  https://github.com/Avraham2489/DS_Training.git (fetch)
# upstream  https://github.com/Avraham2489/DS_Training.git (push)
```

---

## Submitting an Exercise

### Step 1 — Work on the notebook

Open the exercise file (e.g. `D - ML/regression/exercise/regression.ipynb`) and fill in your answers.

> Files ending in `- sol` or starting with `____` are solution files — **do not open or edit them.**

### Step 2 — Commit and push

```bash
git add "D - ML/regression/exercise/regression.ipynb"
git commit -m "Module D - Regression: done"
git push origin main
```

### Step 3 — Open a Pull Request

1. Go to `github.com/<your-username>/DS_Training`
2. Click **Contribute → Open pull request**
3. Set the title using the format in the table below — **this is required**
4. Click **Create pull request**

The instructor will review your work, leave comments on GitHub, and mark a status (needs-revision / complete / submitted).

---

## PR Title Format

Every Pull Request title must follow this exact format: `[your-username] Module X - Topic`

| Module | PR title example |
|---|---|
| B - Python Basics | `[alice123] Module B - Python Basics` |
| B - Python Advanced | `[alice123] Module B - Python Advanced` |
| B - NumPy | `[alice123] Module B - NumPy` |
| B - Pandas | `[alice123] Module B - Pandas` |
| C - Statistics | `[alice123] Module C` |
| D - Regression | `[alice123] Module D - Regression` |
| D - Supervised | `[alice123] Module D - Supervised` |
| D - Unsupervised | `[alice123] Module D - Unsupervised` |
| Optional - Visualization | `[alice123] Optional - Visualization` |

---

## Getting New Material

When the instructor pushes new material, sync your fork:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

> If you have uncommitted local changes, run `git stash` first and `git stash pop` after.

---

## Exams

Exams are **written and held in person** — do not submit them through GitHub.

Exam results are marked by the instructor and will appear automatically on the progress dashboard.


*For questions, contact the instructor directly or leave a comment on your PR.*
