"""Stage 2 postprocessing: repair formatting, then collapse looped output.

Reads <input_dir>/stage2_part*.parquet and writes
<output_dir>/stage2_postprocessing_part*.parquet with three added columns:
    stage2_postprocessed_events, stage2_postprocessed_num_events,
    postprocessing_meta_stage2

Lines still malformed after repair are kept and counted in malformed_count.

    python pipeline/stage2_postprocessing/run_stage2_postprocessing.py \
        --input_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage2 \
        --output_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage2_postprocessing
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

from loop_collapse import collapse_loops, count_bullets
from repair_regex import is_canonical, repair_text


def postprocess(text):
    """Repair every line, then collapse a looping tail. Returns (text, meta)."""
    repaired, repair_meta = repair_text(text)
    clean, loop_meta = collapse_loops(repaired)
    malformed = [line for line in clean.splitlines() if not is_canonical(line)]
    return clean, {
        "lines_in": repair_meta["lines_in"],
        "bullet_lines_in": repair_meta["bullet_lines_in"],
        "lines_out": loop_meta["lines_out"],
        "loop_lines_removed": loop_meta["loop_lines_removed"],
        "non_bullet_lines_dropped": repair_meta["non_bullet_lines_dropped"],
        "blank_lines_dropped": repair_meta["blank_lines_dropped"],
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:10],
        "repairs_applied": repair_meta["repairs_applied"],
    }


def main():
    p = argparse.ArgumentParser(description="Stage 2 postprocessing (repair + loop collapse).")
    p.add_argument("--input_dir", required=True, help="Folder with stage2_part*.parquet.")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "stage2_part*.parquet")))
    if not paths:
        sys.exit(f"No stage2_part*.parquet in {args.input_dir}. Run Stage 2 first.")
    os.makedirs(args.output_dir, exist_ok=True)

    for path in paths:
        df = pd.read_parquet(path)
        texts, metas = [], []
        for events, stage2_meta in zip(df["stage2_events"], df["stage2_meta"]):
            text, meta = postprocess(events)
            meta["upstream_truncated"] = json.loads(stage2_meta)["compute"].get("truncated")
            texts.append(text)
            metas.append(meta)
        df["stage2_postprocessed_events"] = texts
        df["stage2_postprocessed_num_events"] = [count_bullets(t) for t in texts]
        df["postprocessing_meta_stage2"] = [json.dumps(m, ensure_ascii=False) for m in metas]

        name = os.path.basename(path).replace("stage2_part", "stage2_postprocessing_part")
        out_path = os.path.join(args.output_dir, name)
        df.to_parquet(out_path, index=False)
        malformed = sum(m["malformed_count"] for m in metas)
        removed = sum(m["loop_lines_removed"] for m in metas)
        print(f"Wrote {out_path} ({len(df)} rows, {malformed} malformed lines kept, "
              f"{removed} looped lines removed)")


if __name__ == "__main__":
    main()
