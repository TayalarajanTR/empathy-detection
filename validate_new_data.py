#!/usr/bin/env python3
import argparse, csv, json, re, unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import os

REQUIRED_COLS = ["sp_id","rp_id","seeker_post","response_post","level","rationales"]

TASK_ALLOWED = {
    "ER": {2},        # Strong only
    "IP": {1},        # Weak only
    "EX": {1, 2},     # Weak or Strong
}

def norm_text(t: str) -> str:
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t.strip())
    return t

def parse_level(x: str):
    try:
        v = int(x)
        return v if v in {0,1,2} else None
    except Exception:
        return None

def split_rationales(r: str) -> List[str]:
    r = r or ""
    parts = [norm_text(p) for p in r.split("|")]
    return [p for p in parts if p]

def rationale_spans_exist(resp: str, rats: List[str]) -> bool:
    """Every provided rationale must be an exact substring (case-sensitive) of response_post."""
    # Use normalized response to reduce trivial whitespace issues
    resp_n = norm_text(resp)
    for r in rats:
        if r and norm_text(r) not in resp_n:
            return False
    return True

def validate_row(row: Dict[str,str], task: str, allowed_levels: Set[int]) -> Tuple[bool, List[str], Dict[str,str]]:
    reasons = []
    out = {k: row.get(k, "") for k in REQUIRED_COLS}

    # Column presence & basic fields
    for c in REQUIRED_COLS:
        if c not in row:
            reasons.append(f"missing_column:{c}")
    if reasons:
        return False, reasons, out

    # Normalize text fields
    out["sp_id"] = norm_text(out["sp_id"])
    out["rp_id"] = norm_text(out["rp_id"])
    out["seeker_post"] = norm_text(out["seeker_post"])
    out["response_post"] = norm_text(out["response_post"])
    out["rationales"] = row.get("rationales","")  # keep original string; validate separately

    if not out["sp_id"]:
        reasons.append("empty_sp_id")
    if not out["rp_id"]:
        reasons.append("empty_rp_id")
    if not out["seeker_post"]:
        reasons.append("empty_seeker_post")
    if not out["response_post"]:
        reasons.append("empty_response_post")

    # Level check
    lvl = parse_level(row.get("level",""))
    if lvl is None:
        reasons.append("invalid_level_value")
    else:
        out["level"] = str(lvl)
        if lvl not in allowed_levels:
            reasons.append(f"level_not_allowed_for_task:{task}")

    # Rationale check (if provided)
    rats = split_rationales(row.get("rationales",""))
    if rats and not rationale_spans_exist(out["response_post"], rats):
        reasons.append("rationale_span_not_found_in_response")

    is_valid = (len(reasons) == 0)
    return is_valid, reasons, out

def main():
    ap = argparse.ArgumentParser("Validate & filter new empathy data")
    ap.add_argument("--task", required=True, choices=["ER","IP","EX"], help="Which subtask this CSV belongs to")
    ap.add_argument("--input", required=True, help="Path to input CSV (synthetic new data)")
    ap.add_argument("--valid", required=True, help="Where to write valid rows CSV")
    ap.add_argument("--invalid", required=True, help="Where to write invalid rows CSV (with reasons)")
    ap.add_argument("--report", required=True, help="Where to write validation report JSON")
    ap.add_argument("--output", required=True, help="Final output CSV (same as --valid)")
    ap.add_argument("--dedup", action="store_true", help="Drop duplicate (seeker_post,response_post) pairs")
    args = ap.parse_args()

    allowed = TASK_ALLOWED[args.task]

    # Read input
    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"ERROR: input missing required columns: {missing}")
        for r in reader:
            rows.append(r)

    # Validate
    valid_rows, invalid_rows = [], []
    reasons_counter = Counter()
    for r in rows:
        ok, reasons, cleaned = validate_row(r, args.task, allowed)
        if ok:
            valid_rows.append(cleaned)
        else:
            inv = dict(cleaned)
            inv["reasons"] = "|".join(reasons)
            invalid_rows.append(inv)
            reasons_counter.update(reasons)

    # Optional de-dup on (seeker_post,response_post)
    if args.dedup:
        seen = set()
        deduped = []
        dups = 0
        for r in valid_rows:
            key = (r["seeker_post"], r["response_post"])
            if key in seen:
                dups += 1
                continue
            seen.add(key)
            deduped.append(r)
        valid_rows = deduped
        if dups:
            reasons_counter.update({"deduplicated_pairs": dups})

    # Write valid
    with open(args.valid, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLS)
        w.writeheader()
        for r in valid_rows:
            w.writerow(r)

    # Write invalid (with reasons)
    with open(args.invalid, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLS + ["reasons"])
        w.writeheader()
        for r in invalid_rows:
            w.writerow(r)

    # Write report
    report = {
        "task": args.task,
        "allowed_levels": sorted(list(allowed)),
        "input_rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "failure_counts": dict(reasons_counter),
        "notes": [
            "ER keeps only Strong (2).",
            "IP keeps only Weak (1).",
            "EX keeps only Weak (1) and Strong (2).",
            "If rationales provided, each must be an exact substring of response_post.",
            "Use --dedup to drop duplicate (seeker_post,response_post) pairs."
        ]
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    need_header = (not os.path.exists(args.output)) or (os.path.getsize(args.output) == 0)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLS)
        if need_header:
            w.writeheader()
        for r in valid_rows:
            w.writerow(r)

    print(f"[{args.task}] input={len(rows)} valid={len(valid_rows)} invalid={len(invalid_rows)}")
    if reasons_counter:
        print("Top failure reasons:", reasons_counter.most_common(5))

if __name__ == "__main__":
    main()
