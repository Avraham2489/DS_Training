"""
Generates docs/index.html — the trainee progress dashboard.
Reads:  trainees.csv, exams_results.csv, manual_completions.csv, scripts/exercises.json
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
REPO           = "Avraham2489/DS_Training"
ROOT           = Path(__file__).parent.parent
TRAINEES_CSV   = ROOT / "trainees.csv"
EXAMS_CSV      = ROOT / "exams_results.csv"
MANUAL_CSV     = ROOT / "manual_completions.csv"
EXERCISES_JSON = ROOT / "scripts" / "exercises.json"
OUTPUT_HTML    = ROOT / "docs" / "index.html"

STATUS_ORDER = ["complete", "needs-revision", "submitted", ""]

STATUS_STYLE = {
    "complete":       ("✔", "#d4edda", "#155724"),
    "needs-revision": ("↩", "#f8d7da", "#721c24"),
    "submitted":      ("⏳", "#cce5ff", "#004085"),
    "pass":           ("✔", "#d4edda", "#155724"),
    "fail":           ("✘", "#f8d7da", "#721c24"),
    "":               ("—", "#f8f9fa", "#9e9e9e"),
}

SECTION_COLORS = {
    "A":        {"bg": "#1565c0", "light": "#e3f2fd"},
    "B":        {"bg": "#6a1b9a", "light": "#f3e5f5"},
    "C":        {"bg": "#00695c", "light": "#e0f2f1"},
    "D":        {"bg": "#e65100", "light": "#fff3e0"},
    "Optional": {"bg": "#546e7a", "light": "#eceff1"},
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


def cell_html(status: str, is_exam: bool = False) -> str:
    icon, bg, fg = STATUS_STYLE.get(status, STATUS_STYLE[""])
    border_left = "border-left:2px solid #ccc;" if is_exam else ""
    return (
        f'<td style="background:{bg};color:{fg};text-align:center;'
        f'font-weight:bold;padding:10px 0;font-size:15px;{border_left}">'
        f'{icon}</td>'
    )


def progress_bar_html(pct: int, done: int, total: int) -> str:
    if pct >= 80:
        color = "#2e7d32"
    elif pct >= 40:
        color = "#f57f17"
    else:
        color = "#c62828"
    return (
        f'<div style="margin-top:8px">'
        f'<div style="background:#e0e0e0;border-radius:6px;height:10px;width:110px">'
        f'<div style="width:{pct}%;background:{color};height:10px;border-radius:6px"></div>'
        f'</div>'
        f'<div style="font-size:11px;color:#666;margin-top:3px">'
        f'{done}/{total} &nbsp;·&nbsp; <b style="color:{color}">{pct}%</b>'
        f'</div>'
        f'</div>'
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

    manual_raw = load_csv(MANUAL_CSV) if MANUAL_CSV.exists() else []
    manual_overrides: dict[str, dict[str, str]] = {}
    for row in manual_raw:
        manual_overrides.setdefault(row["github_username"], {})[row["key"]] = row["status"].strip()

    exercises: list[dict] = json.loads(EXERCISES_JSON.read_text(encoding="utf-8"))

    # fetch all PRs once
    pr_list = []
    if token:
        try:
            pr_list = gh_get(f"/repos/{REPO}/pulls", token)
        except Exception as exc:
            print(f"Warning: could not fetch PRs — {exc}", file=sys.stderr)

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

    # ── build section groups (preserving order) ───────────────────────────────
    seen_sections: list[str] = []
    section_map: dict[str, dict] = {}
    for ex in exercises:
        sec = ex["section"]
        if sec not in section_map:
            seen_sections.append(sec)
            section_map[sec] = {
                "label":  ex["section_label"],
                "colors": SECTION_COLORS.get(sec, SECTION_COLORS["Optional"]),
                "items":  [],
            }
        section_map[sec]["items"].append(ex)

    countable = [ex for ex in exercises if not ex["optional"]]
    total_items = len(countable)

    # ── header row 1: section spans ───────────────────────────────────────────
    hdr1 = [
        '<th rowspan="2" style="padding:12px 16px;text-align:left;'
        'background:#263238;color:white;min-width:170px;vertical-align:bottom">'
        'Trainee</th>'
    ]
    for sec in seen_sections:
        s = section_map[sec]
        n = len(s["items"])
        bg = s["colors"]["bg"]
        hdr1.append(
            f'<th colspan="{n}" style="padding:8px 12px;text-align:center;'
            f'background:{bg};color:white;border-left:3px solid white;'
            f'font-size:13px;letter-spacing:.5px">'
            f'{s["label"]}</th>'
        )
    hdr1.append(
        '<th rowspan="2" style="padding:12px 16px;text-align:center;'
        'background:#263238;color:white;min-width:120px;border-left:3px solid white;'
        'vertical-align:bottom">Progress</th>'
    )

    # ── header row 2: individual column names ─────────────────────────────────
    hdr2 = []
    for sec in seen_sections:
        s = section_map[sec]
        light = s["colors"]["light"]
        bg    = s["colors"]["bg"]
        items = s["items"]
        for i, ex in enumerate(items):
            is_exam = ex["type"] == "exam"
            border  = f"border-left:2px solid {bg};" if is_exam else ""
            th_style = (
                f"width:40px;vertical-align:bottom;padding:0 0 6px 0;"
                f"background:{light};{border}"
            )
            span_style = (
                f"display:inline-block;writing-mode:vertical-rl;"
                f"transform:rotate(180deg);font-size:11px;font-weight:600;"
                f"color:#333;white-space:nowrap;padding:4px 2px;"
            )
            label = f"📝 {ex['label']}" if is_exam else ex["label"]
            if ex["optional"] and not is_exam:
                label += " (opt)"
            hdr2.append(f'<th style="{th_style}"><span style="{span_style}">{label}</span></th>')

    # ── data rows ─────────────────────────────────────────────────────────────
    rows_html = []
    for t in trainees:
        username  = t["github_username"]
        name      = t["full_name"]
        user_prs  = prs_by_user.get(username, [])
        exam_row  = exams_by_user.get(username, {})
        user_manual = manual_overrides.get(username, {})

        done = 0
        cells = []

        for sec in seen_sections:
            for ex in section_map[sec]["items"]:
                is_exam = ex["type"] == "exam"

                if ex["key"] in user_manual:
                    status = user_manual[ex["key"]]
                elif is_exam:
                    status = exam_row.get(ex["key"], "").strip()
                else:
                    kw = ex["pr_keyword"].lower()
                    matched = [s for title, s in user_prs if kw in title]
                    status = min(matched, key=lambda s: STATUS_ORDER.index(s)) if matched else ""

                cells.append(cell_html(status, is_exam))

                if not ex["optional"] and status in ("complete", "pass"):
                    done += 1

        pct = round(done / total_items * 100) if total_items else 0

        trainee_cell = (
            f'<td style="padding:10px 16px;white-space:nowrap;background:white;'
            f'vertical-align:top">'
            f'<div style="font-size:14px;font-weight:600">{name}</div>'
            f'<div style="font-size:12px;color:#888;margin-top:2px">@{username}</div>'
            f'</td>'
        )
        progress_cell = (
            f'<td style="padding:10px 16px;text-align:center;background:white;'
            f'border-left:3px solid #e0e0e0;vertical-align:middle">'
            f'{progress_bar_html(pct, done, total_items)}'
            f'</td>'
        )

        rows_html.append(
            f"<tr>{trainee_cell}{''.join(cells)}{progress_cell}</tr>"
        )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_entries = [
        ("✔ complete / pass",  "#d4edda", "#155724"),
        ("↩ needs revision",   "#f8d7da", "#721c24"),
        ("⏳ submitted",        "#cce5ff", "#004085"),
        ("✘ fail",             "#f8d7da", "#721c24"),
        ("— not submitted",    "#f8f9fa", "#9e9e9e"),
    ]
    legend_html = "".join(
        f'<span style="background:{bg};color:{fg};padding:4px 12px;'
        f'border-radius:6px;font-size:12px;white-space:nowrap">{text}</span>'
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
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      color: #222;
    }}
    .page-header {{
      background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
      color: white;
      padding: 28px 36px;
      box-shadow: 0 2px 8px rgba(0,0,0,.2);
    }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: .3px; }}
    .page-header .subtitle {{ font-size: 13px; opacity: .65; margin-top: 5px; }}
    .content {{ padding: 28px 36px; }}
    .table-wrapper {{
      overflow-x: auto;
      border-radius: 10px;
      box-shadow: 0 2px 16px rgba(0,0,0,.1);
    }}
    table {{ border-collapse: collapse; background: white; width: 100%; min-width: 0; }}
    th, td {{ border: 1px solid #e0e0e0; }}
    tbody tr {{ transition: background .15s; }}
    tbody tr:hover td {{ filter: brightness(0.96); }}
    .legend {{
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .legend-title {{
      font-size: 12px;
      color: #777;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .5px;
    }}
  </style>
</head>
<body>
  <div class="page-header">
    <h1>DS Training &mdash; Progress Dashboard</h1>
    <div class="subtitle">Updated: {updated}</div>
  </div>
  <div class="content">
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>{"".join(hdr1)}</tr>
          <tr>{"".join(hdr2)}</tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    <div class="legend">
      <span class="legend-title">Legend:</span>
      {legend_html}
    </div>
  </div>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
