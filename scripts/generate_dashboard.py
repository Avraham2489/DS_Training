"""
Generates docs/index.html — the trainee progress dashboard.
Reads:  trainees.csv, exams_results.csv, scripts/exercises.json
Reads:  GitHub PRs via the API (GITHUB_TOKEN env var, injected by Actions)
Writes: docs/index.html
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# ── config ────────────────────────────────────────────────────────────────────
REPO = "Avraham2489/DS_Training"
ROOT = Path(__file__).parent.parent
TRAINEES_CSV = ROOT / "trainees.csv"
EXAMS_CSV = ROOT / "exams_results.csv"
EXERCISES_JSON = ROOT / "scripts" / "exercises.json"
OUTPUT_HTML = ROOT / "docs" / "index.html"

STATUS_ORDER = ["complete", "needs-revision", "submitted", ""]
STATUS_STYLE = {
    "complete":      ("✔", "#28a745", "#22863a"),
    "needs-revision":("↩", "#ffc1cc", "#cb2431"),
    "submitted":     ("⏳", "#0366d6", "#24292e"),
    "pass":          ("✔", "#28a745", "#22863a"),
    "fail":          ("✘", "#ffc1cc", "#cb2431"),
    "":              ("—",  "#f2f2f2", "#999999"),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def gh_get(path: str, token: str) -> list | dict:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    results = []
    url = f"https://api.github.com{path}?per_page=100&state=all"
    while url:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            results.extend(data)
            url = r.links.get("next", {}).get("url")
        else:
            return data
    return results


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(filter(lambda l: not l.startswith("#"), f)):
            rows.append(row)
    return rows


def cell_html(status: str) -> str:
    icon, bg, fg = STATUS_STYLE.get(status, STATUS_STYLE[""])
    return (
        f'<td style="background:{bg};color:{fg};text-align:center;'
        f'font-weight:bold;padding:6px 10px">{icon}</td>'
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("GITHUB_TOKEN not set — skipping PR fetch, showing exams only.", file=sys.stderr)

    trainees = load_csv(TRAINEES_CSV)
    if not trainees:
        print("trainees.csv is empty — nothing to render.", file=sys.stderr)
        sys.exit(0)

    exams_raw = load_csv(EXAMS_CSV)
    exams_by_user = {r["github_username"]: r for r in exams_raw}

    exercises: list[dict] = json.loads(EXERCISES_JSON.read_text(encoding="utf-8"))
    ex_exercises = [e for e in exercises if e["type"] == "exercise"]
    ex_exams     = [e for e in exercises if e["type"] == "exam"]

    # fetch all PRs once
    pr_list = []
    if token:
        try:
            pr_list = gh_get(f"/repos/{REPO}/pulls", token)
        except Exception as exc:
            print(f"Warning: could not fetch PRs — {exc}", file=sys.stderr)

    # index PRs by author → list of (title_lower, top_label)
    def top_label(pr) -> str:
        labels = [l["name"] for l in pr.get("labels", [])]
        for s in STATUS_ORDER:
            if s in labels:
                return s
        if pr.get("state") == "open":
            return "submitted"
        return ""

    prs_by_user: dict[str, list[tuple[str, str]]] = {}
    for pr in pr_list:
        user = pr["user"]["login"]
        prs_by_user.setdefault(user, []).append((pr["title"].lower(), top_label(pr)))

    # ── build rows ────────────────────────────────────────────────────────────
    rows_html = []
    for t in trainees:
        username = t["github_username"]
        name     = t["full_name"]
        user_prs = prs_by_user.get(username, [])
        exam_row = exams_by_user.get(username, {})

        cells = [f'<td style="padding:6px 12px;white-space:nowrap"><b>{name}</b><br>'
                 f'<small style="color:#666">@{username}</small></td>']

        for ex in ex_exercises:
            kw = ex["pr_keyword"].lower()
            matched = [status for title, status in user_prs if kw in title]
            status = min(matched, key=lambda s: STATUS_ORDER.index(s)) if matched else ""
            cells.append(cell_html(status))

        for ex in ex_exams:
            status = exam_row.get(ex["key"], "").strip()
            cells.append(cell_html(status))

        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # ── header ────────────────────────────────────────────────────────────────
    header_cells = ['<th style="padding:8px 12px;text-align:left">Trainee</th>']
    for ex in ex_exercises:
        header_cells.append(
            f'<th style="padding:8px 12px;white-space:nowrap">{ex["label"]}</th>'
        )
    for ex in ex_exams:
        header_cells.append(
            f'<th style="padding:8px 12px;white-space:nowrap;background:#e8eaf6">{ex["label"]}</th>'
        )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_entries = [
        ("✔ complete / pass",  "#28a745", "#22863a"),
        ("↩ needs revision",   "#ffc1cc", "#cb2431"),
        ("⏳ submitted",        "#0366d6", "#24292e"),
        ("✘ fail",             "#ffc1cc", "#cb2431"),
        ("— not submitted",    "#f2f2f2", "#999999"),
    ]
    legend_html = " &nbsp; ".join(
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:4px;font-size:13px">{text}</span>'
        for text, bg, fg in legend_entries
    )

    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="he" dir="ltr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DS Training — Progress Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
    h1   {{ margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    th {{ background: #3f51b5; color: white; }}
    tr:nth-child(even) td:first-child {{ background: #f5f5f5; }}
    td, th {{ border: 1px solid #ddd; }}
    .legend {{ margin-top: 18px; }}
  </style>
</head>
<body>
  <h1>DS Training — Progress Dashboard</h1>
  <div class="subtitle">Updated: {updated}</div>
  <table>
    <thead><tr>{"".join(header_cells)}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
  <div class="legend">{legend_html}</div>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
