#!/usr/bin/env python3
"""DPO Pairs Audio Comparison Viewer Server"""

import json
import os
import re
from flask import Flask, jsonify, request, send_file, abort

app = Flask(__name__)

JSONL_PATH = os.path.join(os.path.dirname(__file__), "..", "kefu_dpo_pairs.jsonl")

# Cache all data in memory on startup
data_cache = []


def load_data():
    global data_cache
    if data_cache:
        return data_cache
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_cache.append(json.loads(line))
    return data_cache


def extract_chosen_vs_rejected_analysis(meta):
    """Extract analysis entries that directly compare the chosen vs rejected versions."""
    history = meta.get("gemini_eval_history", [])
    analyses = []
    for batch in history:
        batch_versions = set()
        for r in batch.get("results", []):
            batch_versions.add(r["version"])
        analyses.append({
            "results": batch["results"],
            "ranking": batch.get("ranking", []),
            "best": batch.get("best", ""),
            "worst": batch.get("worst", ""),
        })
    return analyses


def clean_prompt(text):
    """Clean up spk tags from prompt text."""
    return re.sub(r"<\|spk_\d+\|>", "", text)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/pairs")
def get_pairs():
    load_data()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    search = request.args.get("search", "").strip()

    filtered = data_cache
    if search:
        filtered = [
            d
            for d in data_cache
            if search in d["group_id"] or search in d.get("utt", "")
        ]

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered[start:end]

    items = []
    for d in page_data:
        meta = d.get("meta", {})
        eval_batches = extract_chosen_vs_rejected_analysis(meta)
        items.append(
            {
                "group_id": d["group_id"],
                "utt": d["utt"],
                "prompt": d["prompt"],
                "prompt_clean": clean_prompt(d["prompt"]),
                "chosen": {
                    "version": d["chosen"]["version"],
                    "utt": d["chosen"]["utt"],
                    "text": d["chosen"]["text"],
                    "wav": f"/audio/{d['chosen']['utt']}",
                    "token_count": len(d["chosen"]["token"]),
                },
                "rejected": {
                    "version": d["rejected"]["version"],
                    "utt": d["rejected"]["utt"],
                    "text": d["rejected"]["text"],
                    "wav": f"/audio/{d['rejected']['utt']}",
                    "token_count": len(d["rejected"]["token"]),
                },
                "eval_batches": eval_batches,
                "num_candidates": meta.get("num_candidates", 0),
                "judge_model": meta.get("judge_model", ""),
            }
        )

    return jsonify(
        {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        }
    )


@app.route("/api/pair/<group_id>")
def get_pair(group_id):
    """Get a single pair by group_id."""
    load_data()
    for d in data_cache:
        if d["group_id"] == group_id:
            meta = d.get("meta", {})
            eval_batches = extract_chosen_vs_rejected_analysis(meta)
            return jsonify(
                {
                    "group_id": d["group_id"],
                    "utt": d["utt"],
                    "prompt": d["prompt"],
                    "prompt_clean": clean_prompt(d["prompt"]),
                    "chosen": {
                        "version": d["chosen"]["version"],
                        "utt": d["chosen"]["utt"],
                        "text": d["chosen"]["text"],
                        "wav": f"/audio/{d['chosen']['utt']}",
                        "token_count": len(d["chosen"]["token"]),
                    },
                    "rejected": {
                        "version": d["rejected"]["version"],
                        "utt": d["rejected"]["utt"],
                        "text": d["rejected"]["text"],
                        "wav": f"/audio/{d['rejected']['utt']}",
                        "token_count": len(d["rejected"]["token"]),
                    },
                    "eval_batches": eval_batches,
                    "num_candidates": meta.get("num_candidates", 0),
                    "judge_model": meta.get("judge_model", ""),
                }
            )
    abort(404)


@app.route("/audio/<utt_name>")
def serve_audio(utt_name):
    """Serve audio file by utt name."""
    base_dir = "/home/work_nfs22/xmren/data/kefu/90w_tokens_noseed"
    wav_path = os.path.join(base_dir, utt_name + ".wav")
    if not os.path.exists(wav_path):
        abort(404, f"Audio file not found: {wav_path}")
    return send_file(wav_path, mimetype="audio/wav")


@app.route("/api/stats")
def get_stats():
    load_data()
    versions_chosen = {}
    versions_rejected = {}
    for d in data_cache:
        cv = d["chosen"]["version"]
        rv = d["rejected"]["version"]
        versions_chosen[cv] = versions_chosen.get(cv, 0) + 1
        versions_rejected[rv] = versions_rejected.get(rv, 0) + 1
    return jsonify(
        {
            "total_pairs": len(data_cache),
            "versions_chosen": dict(
                sorted(versions_chosen.items(), key=lambda x: -x[1])
            ),
            "versions_rejected": dict(
                sorted(versions_rejected.items(), key=lambda x: -x[1])
            ),
        }
    )


if __name__ == "__main__":
    import subprocess

    print(f"Loading data from: {JSONL_PATH}")
    load_data()
    print(f"Loaded {len(data_cache)} DPO pairs")

    # detect public IPv6
    try:
        out = subprocess.check_output(
            "ip -6 addr show ppp0 | grep 'scope global' | grep -v deprecated | head -1 | awk '{print $2}' | cut -d/ -f1",
            shell=True, text=True
        ).strip()
        if out:
            print(f"Public IPv6: http://[{out}]:8765")
    except Exception:
        pass

    # detect LAN IPv4
    try:
        out = subprocess.check_output(
            "hostname -I 2>/dev/null | awk '{print $1}'",
            shell=True, text=True
        ).strip()
        if out and out != "127.0.0.1":
            print(f"LAN    IPv4: http://{out}:8765")
    except Exception:
        pass

    print(f"Server starting at http://0.0.0.0:8765")
    app.run(host="::", port=8765, debug=True)
