"""Helpers for running open-source models with vLLM.

The same file sits in every stage's open_source/ folder so each folder is
self-contained.
"""

import inspect
import json
import os
import time
from datetime import datetime, timezone

from vllm import LLM, SamplingParams

BULLETS = ("- ", "* ", "• ", "· ")


def load_agent_spec(path, agent_name):
    """Return one model entry from an agent_specs.json file."""
    with open(path, encoding="utf-8") as f:
        specs = json.load(f)
    if agent_name not in specs:
        raise KeyError(f"'{agent_name}' not in {path}. Available: {sorted(specs)}")
    return specs[agent_name]


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read().rstrip()


def count_events(text):
    """Number of bullet lines in a model output."""
    return sum(line.lstrip().startswith(BULLETS) for line in (text or "").splitlines())


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def build_meta(agent_name, spec, prompt_version, compute):
    """Provenance saved with every output row."""
    return {
        "agent_name": agent_name,
        "model_id": spec["model"],
        "provider": "vllm",
        "prompt_version": prompt_version,
        "gen_params": spec.get("generation", {}),
        "chat_template_kwargs": spec.get("chat_template_kwargs"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compute": compute,
    }


def write_parquet(df, path):
    """Write through a temp file so a crash never leaves a half-written chunk."""
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build_engine(spec):
    """Load the model. Every key in spec["request"] is passed to vllm.LLM."""
    request = spec.get("request", {})
    print(f"Loading {spec['model']} with {request}")
    return LLM(model=spec["model"], **request)


def build_sampling_params(generation):
    """Turn spec["generation"] into SamplingParams, rejecting unknown keys."""
    unknown = set(generation) - set(inspect.signature(SamplingParams).parameters)
    if unknown:
        raise ValueError(f"Unknown sampling params in agent spec: {sorted(unknown)}")
    return SamplingParams(**generation)


def generate(llm, conversations, sampling_params, chat_template_kwargs=None):
    """Run all conversations as one vLLM batch.

    Returns one {"text", "error", "compute"} dict per conversation, in input order.
    chat_template_kwargs is for models such as Qwen3.6 that need
    {"enable_thinking": false}.
    """
    n = len(conversations)
    # Longest prompts first so they don't hold up the end of the batch.
    order = sorted(range(n), reverse=True,
                   key=lambda i: sum(len(m["content"]) for m in conversations[i]))
    start = time.time()
    try:
        outputs = llm.chat([conversations[i] for i in order], sampling_params=sampling_params,
                           chat_template_kwargs=chat_template_kwargs)
    except Exception as e:
        print(f"Batch failed: {e}")
        return [{"text": "", "error": f"batch failure: {e}", "compute": {}} for _ in range(n)]
    seconds = round((time.time() - start) / max(n, 1), 3)  # batch time split evenly

    results = [None] * n
    for i, out in zip(order, outputs):
        completion = out.outputs[0]
        results[i] = {
            "text": completion.text or "",
            "error": None,
            "compute": {
                "input_tokens": len(out.prompt_token_ids or []),
                "output_tokens": len(completion.token_ids or []),
                "generation_time_sec": seconds,
                "truncated": completion.finish_reason == "length",
            },
        }
    return results
