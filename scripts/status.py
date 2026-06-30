#!/usr/bin/env python3
"""
Lightweight trainee status updater — set a module's status and advance trainees.

Two modes:

  # interactive — pick a trainee, see their track with current statuses, mark modules:
  python scripts/status.py
  python scripts/status.py <username>

  # one-shot:
  python scripts/status.py <username> <module_key> <status>

Statuses:
  exercises / assignments : complete | needs-revision | submitted | clear
  exams                   : pass | fail | clear        (keys starting with "exam_")

Where it writes:
  exams                   -> exams_results.csv      (one column per exam)
  exercises / assignments -> manual_completions.csv (override on top of the PR label)
  "clear" removes the override / empties the exam cell.

After updating, the tool commits and pushes to main, so the CI rebuilds the dashboard.
Run with --no-push to only edit the CSVs locally.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
TRAINEES_CSV   = ROOT / "trainees.csv"
EXAMS_CSV      = ROOT / "exams_results.csv"
MANUAL_CSV     = ROOT / "manual_completions.csv"
OVERRIDES_CSV  = ROOT / "path_overrides.csv"
EXERCISES_JSON = ROOT / "scripts" / "exercises.json"
TRACKS_JSON    = ROOT / "scripts" / "tracks.json"

EXERCISE_STATUSES = ["complete", "needs-revision", "submitted", "clear"]
EXAM_STATUSES     = ["pass", "fail", "clear"]

ICON = {"exam": "📝", "assignment": "📋"}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(filter(lambda l: not l.startswith("#"), f)))


def load_world():
    trainees  = load_csv(TRAINEES_CSV)
    exercises = json.loads(EXERCISES_JSON.read_text(encoding="utf-8"))
    tracks    = json.loads(TRACKS_JSON.read_text(encoding="utf-8")) if TRACKS_JSON.exists() else {}
    overrides: dict[str, dict[str, set]] = {}
    for row in load_csv(OVERRIDES_CSV):
        e = overrides.setdefault(row["github_username"], {"add": set(), "remove": set()})
        action = row["action"].strip().lower()
        if action in e:
            e[action].add(row["key"])
    return trainees, exercises, tracks, overrides


def assigned_keys(trainee: dict, exercises: list[dict], tracks: dict, overrides: dict) -> set:
    all_keys = {ex["key"] for ex in exercises}
    track = tracks.get((trainee.get("track") or "").strip())
    keys = set(track["items"]) if track else set(all_keys)
    ov = overrides.get(trainee["github_username"], {})
    keys |= ov.get("add", set())
    keys -= ov.get("remove", set())
    return keys & all_keys


def is_exam(ex: dict) -> bool:
    return ex["type"] == "exam"


# ── current-status lookups (no GitHub call — keeps the tool offline & fast) ──────

def manual_map() -> dict[tuple, str]:
    return {(r["github_username"], r["key"]): r["status"].strip() for r in load_csv(MANUAL_CSV)}


def exam_map() -> dict[str, dict]:
    return {r["github_username"]: r for r in load_csv(EXAMS_CSV)}


def current_status(username: str, ex: dict, manual: dict, exams: dict) -> str:
    if is_exam(ex):
        return (exams.get(username, {}).get(ex["key"]) or "").strip()
    return manual.get((username, ex["key"]), "")


# ── writers ─────────────────────────────────────────────────────────────────

def set_exam(username: str, key: str, status: str) -> Path:
    rows = load_csv(EXAMS_CSV)
    fields = list(rows[0].keys()) if rows else ["github_username"]
    if key not in fields:
        fields.append(key)
        for r in rows:
            r.setdefault(key, "")
    row = next((r for r in rows if r["github_username"] == username), None)
    if row is None:
        row = {f: "" for f in fields}
        row["github_username"] = username
        rows.append(row)
    row[key] = "" if status == "clear" else status
    with open(EXAMS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return EXAMS_CSV


def set_manual(username: str, key: str, status: str) -> Path:
    rows = [r for r in load_csv(MANUAL_CSV)]
    rows = [r for r in rows if not (r["github_username"] == username and r["key"] == key)]
    if status != "clear":
        rows.append({"github_username": username, "key": key, "status": status})
    with open(MANUAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["github_username", "key", "status"])
        w.writeheader()
        w.writerows(rows)
    return MANUAL_CSV


def apply_status(username: str, ex: dict, status: str) -> Path:
    return set_exam(username, ex["key"], status) if is_exam(ex) else set_manual(username, ex["key"], status)


# ── git ─────────────────────────────────────────────────────────────────────

def commit_push(paths: set, message: str, push: bool):
    rel = sorted(str(p) for p in paths)
    subprocess.run(["git", "-C", str(ROOT), "add", *rel], check=True)
    res = subprocess.run(["git", "-C", str(ROOT), "commit", "-m", message],
                         capture_output=True, text=True)
    if res.returncode != 0:
        if "nothing to commit" in (res.stdout + res.stderr):
            print("Nothing changed — no commit made.")
            return
        print(res.stdout + res.stderr, file=sys.stderr)
        sys.exit(1)
    print(res.stdout.strip())
    if push:
        subprocess.run(["git", "-C", str(ROOT), "push", "origin", "main"], check=True)
        print("Pushed to origin/main — dashboard will refresh via CI.")
    else:
        print("Committed locally (--no-push). Run the dashboard or push when ready.")


# ── helpers ─────────────────────────────────────────────────────────────────

def find_trainee(trainees: list[dict], name: str) -> dict | None:
    name = name.strip().lower()
    for t in trainees:
        if t["github_username"].lower() == name:
            return t
    return None


def valid_statuses(ex: dict) -> list[str]:
    return EXAM_STATUSES if is_exam(ex) else EXERCISE_STATUSES


# ── interactive mode ──────────────────────────────────────────────────────────

def choose_trainee(trainees: list[dict]) -> dict:
    print("\nTrainees:")
    for i, t in enumerate(trainees, 1):
        print(f"  {i}) {t['full_name']:20} @{t['github_username']:15} [{t.get('track') or 'full'}]")
    while True:
        sel = input("\nPick a trainee (number): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(trainees):
            return trainees[int(sel) - 1]
        print("  invalid choice.")


def interactive(trainee: dict, exercises: list[dict], tracks: dict, overrides: dict, push: bool):
    keys = assigned_keys(trainee, exercises, tracks, overrides)
    track_items = [ex for ex in exercises if ex["key"] in keys]
    username = trainee["github_username"]
    track_label = tracks.get((trainee.get("track") or "").strip(), {}).get("label", trainee.get("track") or "full")
    changed: set = set()
    summary: list[str] = []

    while True:
        manual, exams = manual_map(), exam_map()
        print(f"\n=== {trainee['full_name']} (@{username}) — track: {track_label} ===")
        for i, ex in enumerate(track_items, 1):
            st = current_status(username, ex, manual, exams) or "—"
            tag = ICON.get(ex["type"], "  ")
            print(f"  {i:2}) {tag} {ex['label']:22} ({ex['key']:20}) : {st}")
        sel = input("\nModule number to update (Enter to finish): ").strip()
        if sel == "":
            break
        if not (sel.isdigit() and 1 <= int(sel) <= len(track_items)):
            print("  invalid choice.")
            continue
        ex = track_items[int(sel) - 1]
        opts = valid_statuses(ex)
        print("  status: " + "  ".join(f"[{j}] {s}" for j, s in enumerate(opts, 1)))
        ans = input("  choose (number or text): ").strip().lower()
        status = opts[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(opts) else ans
        if status not in opts:
            print(f"  invalid status (allowed: {', '.join(opts)}).")
            continue
        changed.add(apply_status(username, ex, status))
        summary.append(f"{ex['key']}={status}")
        print(f"  ✓ {ex['label']} → {status}")

    if not changed:
        print("\nNo changes made.")
        return
    commit_push(changed, f"chore: update {username} — {', '.join(summary)}", push)


# ── one-shot mode ──────────────────────────────────────────────────────────────

def one_shot(username: str, key: str, status: str, push: bool):
    trainees, exercises, tracks, overrides = load_world()
    trainee = find_trainee(trainees, username)
    if trainee is None:
        sys.exit(f"Unknown trainee '{username}'. Known: " +
                 ", ".join(t["github_username"] for t in trainees))
    ex = next((e for e in exercises if e["key"] == key), None)
    if ex is None:
        sys.exit(f"Unknown module key '{key}'. Known: " +
                 ", ".join(e["key"] for e in exercises))
    status = status.strip().lower()
    if status not in valid_statuses(ex):
        sys.exit(f"Invalid status '{status}' for {key}. Allowed: {', '.join(valid_statuses(ex))}")
    path = apply_status(trainee["github_username"], ex, status)
    print(f"Set {trainee['github_username']} · {ex['label']} ({key}) → {status}")
    commit_push({path}, f"chore: update {trainee['github_username']} — {key}={status}", push)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if a != "--no-push"]
    push = "--no-push" not in sys.argv[1:]
    trainees, exercises, tracks, overrides = load_world()

    if len(args) >= 3:
        one_shot(args[0], args[1], args[2], push)
    elif len(args) == 1:
        t = find_trainee(trainees, args[0])
        if t is None:
            sys.exit(f"Unknown trainee '{args[0]}'.")
        interactive(t, exercises, tracks, overrides, push)
    elif len(args) == 0:
        if not trainees:
            sys.exit("trainees.csv is empty.")
        interactive(choose_trainee(trainees), exercises, tracks, overrides, push)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
