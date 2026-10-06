"""Stage 2: temporal tagging with an open-source model (vLLM).

For each patient, gives the model the anchor dates, the discharge summary and
the postprocessed Stage 1 events, and asks it to prefix every event with a date
and one tag: [EXACT], [APPROX], [PRE ADM] or [INDETERMINATE].

Reads <input_dir>/stage1_part*.parquet (Stage 1 postprocessing output) and
writes <output_dir>/stage2_part*.parquet with four added columns:
    stage2_events, stage2_num_events, stage2_meta, stage2_error

    python pipeline/stage2_time_tagging/open_source/time_tagging.py \
        --input_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage1_postprocessing \
        --output_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage2 \
        --agent_name qwen3_30b_a3b_instruct_2507
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

from llm_utils import (build_engine, build_meta, build_sampling_params, count_events, generate,
                       load_agent_spec, load_text, to_json, write_parquet)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "generation_prompts"
EVENTS_COLUMN = "stage1_postprocessed_events"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 temporal tagging (vLLM).")
    p.add_argument("--input_dir", required=True, help="Stage 1 postprocessing folder.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--agent_name", required=True, help="Model key in agent_specs.json.")
    p.add_argument("--agent_specs", default=HERE / "agent_specs.json")
    p.add_argument("--prompt_version", default="v2")
    p.add_argument("--overwrite", action="store_true", help="Redo chunks that already exist.")
    return p.parse_args()


def date_only(value):
    """'2162-08-14 00:00:00' -> '2162-08-14'."""
    return "" if value is None else str(value).strip().split(" ")[0]


def build_messages(system_prompt, user_template, original, events):
    age = original.get("AGE_AT_CHARTDATE")
    user = user_template.format(
        dob=date_only(original.get("DOB")),
        admission_date=date_only(original.get("ADMITTIME")),
        discharge_date=date_only(original.get("DISCHTIME")),
        patient_age="" if age is None else age,
        discharge_summary=original.get("TEXT") or "",
        atomic_clinical_events=events or "",
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def main():
    args = parse_args()
    spec = load_agent_spec(args.agent_specs, args.agent_name)
    system_prompt = load_text(PROMPTS / f"stage2_system_prompt_{args.prompt_version}.txt")
    user_template = load_text(PROMPTS / f"stage2_user_prompt_{args.prompt_version}.txt")

    paths = sorted(glob.glob(os.path.join(args.input_dir, "stage1_part*.parquet")))
    if not paths:
        sys.exit(f"No stage1_part*.parquet in {args.input_dir}. Run Stage 1 postprocessing first.")
    os.makedirs(args.output_dir, exist_ok=True)
    sampling_params = build_sampling_params(spec.get("generation", {}))
    llm = None  # loaded on the first chunk that needs work

    for path in paths:
        name = os.path.basename(path).replace("stage1_part", "stage2_part")
        out_path = os.path.join(args.output_dir, name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipping (use --overwrite to redo)")
            continue

        df = pd.read_parquet(path)
        if EVENTS_COLUMN not in df:
            sys.exit(f"{path} has no {EVENTS_COLUMN} column. Run Stage 1 postprocessing first.")
        conversations = [
            build_messages(system_prompt, user_template, json.loads(original), events)
            for original, events in zip(df["original_data"], df[EVENTS_COLUMN])
        ]
        llm = llm or build_engine(spec)
        results = generate(llm, conversations, sampling_params, spec.get("chat_template_kwargs"))

        df["stage2_events"] = [r["text"] for r in results]
        df["stage2_num_events"] = [count_events(r["text"]) for r in results]
        df["stage2_meta"] = [to_json(build_meta(args.agent_name, spec, args.prompt_version,
                                                r["compute"])) for r in results]
        df["stage2_error"] = [r["error"] for r in results]
        write_parquet(df, out_path)
        failed = sum(r["error"] is not None for r in results)
        print(f"Wrote {out_path} ({len(df)} rows, {failed} failed)")


if __name__ == "__main__":
    main()
