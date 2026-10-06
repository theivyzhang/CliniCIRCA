"""Stage 1 postprocessing: collapse looped output in each Stage 1 event list.

Reads <input_dir>/stage1_part*.parquet and writes the same files to
<output_dir> with three added columns:
    stage1_postprocessed_events, stage1_postprocessed_num_events,
    postprocessing_meta_stage1

    python pipeline/stage1_postprocessing/run_stage1_postprocessing.py \
        --input_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage1 \
        --output_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage1_postprocessing
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

from loop_collapse import collapse_loops, count_bullets


def main():
    p = argparse.ArgumentParser(description="Stage 1 postprocessing (loop collapse).")
    p.add_argument("--input_dir", required=True, help="Folder with stage1_part*.parquet.")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "stage1_part*.parquet")))
    if not paths:
        sys.exit(f"No stage1_part*.parquet in {args.input_dir}. Run Stage 1 first.")
    os.makedirs(args.output_dir, exist_ok=True)

    for path in paths:
        df = pd.read_parquet(path)
        results = [collapse_loops(text) for text in df["stage1_events"]]
        df["stage1_postprocessed_events"] = [text for text, _ in results]
        df["stage1_postprocessed_num_events"] = [count_bullets(text) for text, _ in results]
        df["postprocessing_meta_stage1"] = [json.dumps(meta) for _, meta in results]

        out_path = os.path.join(args.output_dir, os.path.basename(path))
        df.to_parquet(out_path, index=False)
        removed = sum(meta["loop_lines_removed"] for _, meta in results)
        print(f"Wrote {out_path} ({len(df)} rows, {removed} looped lines removed)")


if __name__ == "__main__":
    main()
