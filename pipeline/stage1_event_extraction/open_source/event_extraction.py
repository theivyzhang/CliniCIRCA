"""Stage 1: atomic clinical event extraction with an open-source model (vLLM).

Reads a patient parquet (one discharge summary per row), shows the model one
worked example, and asks it to list every atomic clinical event in the note.
Writes chunked parquet to <output_dir>/stage1_part0001.parquet, ...

    python pipeline/stage1_event_extraction/open_source/event_extraction.py \
        --input_parquet dataset/sample_patient.parquet \
        --output_dir outputs/open_source/qwen3_30b_a3b_instruct_2507/stage1 \
        --agent_name qwen3_30b_a3b_instruct_2507
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from llm_utils import (build_engine, build_meta, build_sampling_params, count_events,
                       generate, load_agent_spec, load_text, to_json, write_parquet)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "generation_prompts"

# Input columns carried forward for the later stages.
ORIGINAL_DATA_COLUMNS = ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "DOB", "ADMITTIME",
                         "DISCHTIME", "AGE_AT_CHARTDATE", "TEXT"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 event extraction (vLLM).")
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--agent_name", required=True, help="Model key in agent_specs.json.")
    p.add_argument("--agent_specs", default=HERE / "agent_specs.json")
    p.add_argument("--prompt_version", default="v1")
    p.add_argument("--chunk_size", type=int, default=200, help="Rows per output file.")
    p.add_argument("--limit", type=int, help="Only process the first N rows.")
    p.add_argument("--overwrite", action="store_true", help="Redo chunks that already exist.")
    return p.parse_args()


def build_messages(system_prompt, user_template, example_note, example_events, note):
    """System prompt, one worked example, then the note to process."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(example_note)},
        {"role": "assistant", "content": example_events},
        {"role": "user", "content": user_template.format(note)},
    ]


def original_data(row):
    """Input fields as plain JSON-ready Python values."""
    out = {}
    for col in ORIGINAL_DATA_COLUMNS:
        value = row.get(col)
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):  # numpy scalar -> Python scalar
            value = value.item()
        out[col] = value
    return out


def main():
    args = parse_args()
    spec = load_agent_spec(args.agent_specs, args.agent_name)
    system_prompt = load_text(PROMPTS / f"stage1_system_prompt_{args.prompt_version}.txt")
    user_template = load_text(PROMPTS / f"stage1_user_prompt_{args.prompt_version}.txt")
    example_note = load_text(PROMPTS / "single_shot_discharge_summary.txt")
    example_events = load_text(PROMPTS / "single_shot_atomic_events.txt")

    df = pd.read_parquet(args.input_parquet).head(args.limit)
    os.makedirs(args.output_dir, exist_ok=True)
    sampling_params = build_sampling_params(spec.get("generation", {}))
    llm = None  # loaded on the first chunk that needs work

    for part, start in enumerate(range(0, len(df), args.chunk_size), start=1):
        out_path = os.path.join(args.output_dir, f"stage1_part{part:04d}.parquet")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipping (use --overwrite to redo)")
            continue

        chunk = df.iloc[start:start + args.chunk_size]
        conversations = [
            build_messages(system_prompt, user_template, example_note, example_events, str(note))
            for note in chunk["TEXT"]
        ]
        llm = llm or build_engine(spec)
        results = generate(llm, conversations, sampling_params, spec.get("chat_template_kwargs"))

        rows = []
        for (_, row), r in zip(chunk.iterrows(), results):
            rows.append({
                "subject_id": str(row["SUBJECT_ID"]),
                "hadm_id": str(row["HADM_ID"]),
                "stage1_events": r["text"],
                "stage1_num_events": count_events(r["text"]),
                "stage1_error": r["error"],
                "original_data": to_json(original_data(row)),
                "stage1_meta": to_json(build_meta(args.agent_name, spec, args.prompt_version,
                                                  r["compute"])),
            })
        write_parquet(pd.DataFrame(rows), out_path)
        failed = sum(r["error"] is not None for r in results)
        print(f"Wrote {out_path} ({len(rows)} rows, {failed} failed)")


if __name__ == "__main__":
    main()
