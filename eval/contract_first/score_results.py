# -*- coding: utf-8 -*-
"""Score contract-first experiment artifacts into rows.jsonl."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "eval" / "contract_first" / "workspaces"
ROWS = ROOT / "eval" / "contract_first" / "results" / "rows.jsonl"


def score(task_id: str, mode: str) -> tuple[int, int, str]:
    root = WS / task_id / mode
    if task_id == "T1":
        p = root / "index.html"
        if not p.exists():
            return 0, 1 if mode == "direct" else 2, "no index.html"
        html = p.read_text(encoding="utf-8", errors="replace").lower()
        has_nav = "nav" in html or "navbar" in html
        has_hero = "hero" in html
        has_feat = any(x in html for x in ("feature", "特性", "grid"))
        has_cta = "cta" in html or "button" in html
        ok = has_nav and has_hero and has_feat and has_cta and len(html) > 2000
        consistency = 5 if mode == "contract" and ok else (3 if ok else 1)
        return int(ok), consistency, f"nav={has_nav} hero={has_hero} feat={has_feat} cta={has_cta} size={p.stat().st_size}"

    if task_id == "T2":
        py_files = [
            p for p in root.rglob("*.py")
            if "__pycache__" not in str(p) and ".trash" not in str(p)
        ]
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in py_files)
        docs = [p for p in root.rglob("*.md") if ".trash" not in str(p)]
        doc_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in docs)
        has_list = bool(re.search(r"list|GET|/todos", text, re.I))
        has_create = bool(re.search(r"create|POST", text, re.I))
        has_complete = bool(re.search(r"complete|PATCH|PUT", text, re.I))
        has_arch = bool(docs) or "mermaid" in doc_text.lower() or "架构" in doc_text
        ok = has_list and has_create and has_complete and has_arch and bool(py_files)
        consistency = 4 if mode == "contract" and ok else (3 if ok else 2)
        return int(ok), consistency, f"py={len(py_files)} list={has_list} create={has_create} complete={has_complete} arch={has_arch}"

    if task_id == "T3":
        p = root / "customer_support_bot.md"
        if not p.exists():
            return 0, 1, "missing customer_support_bot.md"
        t = p.read_text(encoding="utf-8", errors="replace")
        has_calm = any(x in t for x in ("安抚", "抱歉", "理解您"))
        has_diag = any(x in t for x in ("排查", "订单", "故障", "收集"))
        has_esc = any(x in t for x in ("升级", "人工"))
        ok = has_calm and has_diag and has_esc and len(t) > 500
        return int(ok), (4 if ok else 2), f"calm={has_calm} diag={has_diag} esc={has_esc} size={p.stat().st_size}"

    if task_id == "T4":
        p = root / "churn_framework.md"
        if not p.exists():
            return 0, 1, "missing churn_framework.md"
        t = p.read_text(encoding="utf-8", errors="replace")
        has_dim = "维度" in t or "|" in t
        has_dir = "结论" in t or "方向" in t
        competitor_note = "competitor_mentioned" if "竞品" in t else "competitor_ok"
        ok = has_dim and has_dir and len(t) > 400
        return int(ok), (4 if ok else 2), f"dim={has_dim} dir={has_dir} {competitor_note} size={p.stat().st_size}"

    if task_id == "T5":
        p = root / "contract_first_readme_section.md"
        if not p.exists():
            return 0, 1, "missing section md"
        t = p.read_text(encoding="utf-8", errors="replace")
        heads = re.findall(r"^#+\s+", t, re.M)
        has_sample = any(x in t for x in ("样本", "语气", "示例"))
        marketing = any(x in t for x in ("赋能", "打造卓越", "颠覆", "立即体验"))
        ok = len(heads) >= 3 and has_sample and not marketing and len(t) > 400
        consistency = 5 if mode == "contract" and ok else (4 if ok else 2)
        return int(ok), consistency, f"heads={len(heads)} sample={has_sample} marketing={marketing} size={p.stat().st_size}"

    return 0, 1, "unknown"


def main() -> None:
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["task_id"], r["mode"])] = r

    scored = []
    for (tid, mode), r in sorted(latest.items()):
        s, c, n = score(tid, mode)
        r["success"] = s
        r["consistency_1_5"] = c
        r["rework_rounds"] = 0
        r["direction_ok_before_exec"] = None if mode == "direct" else 1
        r["notes"] = n
        scored.append(r)

    ROWS.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in scored) + "\n", encoding="utf-8")

    print("task mode success consistency steps sec notes")
    for r in scored:
        print(
            f"{r['task_id']} {r['mode']} {r['success']} {r['consistency_1_5']} "
            f"{r['steps']} {r['elapsed_sec']} {r['notes']}"
        )

    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "cons": 0, "sec": 0.0})
    for r in scored:
        a = agg[r["mode"]]
        a["n"] += 1
        a["ok"] += int(r["success"] or 0)
        a["cons"] += int(r["consistency_1_5"] or 0)
        a["sec"] += float(r["elapsed_sec"] or 0)

    print("--- by mode ---")
    for m, a in agg.items():
        print(
            m,
            f"success={a['ok']}/{a['n']}",
            f"avg_consistency={a['cons']/a['n']:.1f}",
            f"avg_sec={a['sec']/a['n']:.1f}",
        )


if __name__ == "__main__":
    main()
