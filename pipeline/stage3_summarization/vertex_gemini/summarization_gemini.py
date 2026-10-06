"""Stage 3: prose summary of the tagged timeline with Gemini on Vertex AI.

The postprocessed Stage 2 timeline is the only source of facts and dates; the
original discharge summary is given as context for wording only.

Reads <input_dir>/stage2_postprocessing_part*.parquet and writes
<output_dir>/stage3_part*.parquet with three added columns:
    stage3_summary, stage3_meta, stage3_error

    python pipeline/stage3_summarization/vertex_gemini/summarization_gemini.py \
        --input_dir outputs/vertex_gemini/gemini_2_5_pro/stage2_postprocessing \
        --output_dir outputs/vertex_gemini/gemini_2_5_pro/stage3 \
        --agent_name gemini_2_5_pro --project "PLACEHOLDER: ENTER YOUR PROJECT ID"
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

from llm_utils_gemini import (add_vertex_args, build_client, build_generation_config, build_meta,
                              generate, load_agent_spec, load_text, to_json, write_parquet)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "generation_prompts"
TIMELINE_COLUMN = "stage2_postprocessed_events"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 summarization (Gemini on Vertex AI).")
    p.add_argument("--input_dir", required=True, help="Stage 2 postprocessing folder.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--agent_name", required=True, help="Model key in agent_specs.json.")
    p.add_argument("--agent_specs", default=HERE / "agent_specs.json")
    p.add_argument("--prompt_version", default="v1")
    p.add_argument("--overwrite", action="store_true", help="Redo chunks that already exist.")
    add_vertex_args(p)
    return p.parse_args()


def build_messages(system_prompt, user_template, original, timeline):
    user = user_template.format(discharge_summary=original.get("TEXT") or "",
                                clinical_events_timeline=timeline or "")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def main():
    args = parse_args()
    spec = load_agent_spec(args.agent_specs, args.agent_name)
    system_prompt = load_text(PROMPTS / f"stage3_system_prompt_{args.prompt_version}.txt")
    user_template = load_text(PROMPTS / f"stage3_user_prompt_{args.prompt_version}.txt")

    paths = sorted(glob.glob(os.path.join(args.input_dir, "stage2_postprocessing_part*.parquet")))
    if not paths:
        sys.exit(f"No stage2_postprocessing_part*.parquet in {args.input_dir}. "
                 "Run Stage 2 postprocessing first.")
    os.makedirs(args.output_dir, exist_ok=True)
    client = build_client(args.project, args.location)
    gen_config = build_generation_config(spec["generation"])

    for path in paths:
        name = os.path.basename(path).replace("stage2_postprocessing_part", "stage3_part")
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipping (use --overwrite to redo)")
            continue

        df = pd.read_parquet(path)
        conversations = [
            build_messages(system_prompt, user_template, json.loads(original), timeline)
            for original, timeline in zip(df["original_data"], df[TIMELINE_COLUMN])
        ]
        results = generate(client, spec["model"], conversations, gen_config, args.mode,
                           args.max_workers, args.gcs_bucket,
                           f"{args.gcs_prefix}/stage3/{name.removesuffix('.parquet')}")

        df["stage3_summary"] = [r["text"] for r in results]
        df["stage3_meta"] = [to_json(build_meta(args.agent_name, spec, args.prompt_version,
                                                r["compute"])) for r in results]
        df["stage3_error"] = [r["error"] for r in results]
        write_parquet(df, out_path)
        failed = sum(r["error"] is not None for r in results)
        print(f"Wrote {out_path} ({len(df)} rows, {failed} failed)")


if __name__ == "__main__":
    main()
