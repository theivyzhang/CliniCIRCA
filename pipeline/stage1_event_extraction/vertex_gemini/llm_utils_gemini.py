"""Helpers for running Gemini on Vertex AI.

The same file sits in every stage's vertex_gemini/ folder so each folder is
self-contained.

Two run modes:
  stream  real-time requests sent in parallel (default)
  batch   one Vertex AI batch job per chunk, staged through Cloud Storage;
          half the price of stream but asynchronous (minutes to hours)

Authentication uses Application Default Credentials: run
`gcloud auth application-default login`, or point GOOGLE_APPLICATION_CREDENTIALS
at a service-account key. API keys are not used.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from google import genai
from tqdm import tqdm

BULLETS = ("- ", "* ", "• ", "· ")
BATCH_DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                     "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


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
        "provider": "vertex_ai",
        "prompt_version": prompt_version,
        "gen_params": spec.get("generation", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compute": compute,
    }


def write_parquet(df, path):
    """Write through a temp file so a crash never leaves a half-written chunk."""
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def add_vertex_args(parser):
    """Command-line options shared by the vertex_gemini scripts."""
    g = parser.add_argument_group("Vertex AI")
    g.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                   help="Google Cloud project (default: $GOOGLE_CLOUD_PROJECT).")
    g.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    g.add_argument("--mode", choices=["stream", "batch"], default="stream")
    g.add_argument("--max_workers", type=int, default=8, help="Parallel requests in stream mode.")
    g.add_argument("--gcs_bucket", help="Cloud Storage bucket for batch mode (name only).")
    g.add_argument("--gcs_prefix", default="clinicirca_batch", help="Folder inside the bucket.")


def build_client(project, location):
    if not project or project.startswith("PLACEHOLDER"):
        raise SystemExit("Enter your Google Cloud project ID: --project, GOOGLE_CLOUD_PROJECT, "
                         "or the placeholder at the top of run_pipeline.sh.")
    check_cache_disabled(project, location)
    print(f"Vertex AI project={project} location={location}")
    return genai.Client(vertexai=True, project=project, location=location)


def check_cache_disabled(project, location):
    """Warn unless Vertex AI prompt caching is turned off for the project.

    Data use agreements for real patient data (e.g. PhysioNet's for MIMIC)
    require that the provider keeps no copy of the data. Vertex AI caches
    prompts by default, so check the project setting before every run.
    """
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/cacheConfig"
    try:
        import google.auth
        import google.auth.transport.requests
        import requests

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        resp = requests.get(url, headers={"Authorization": f"Bearer {creds.token}"}, timeout=30)
        resp.raise_for_status()
        disabled = bool(resp.json().get("disableCache"))
    except Exception as e:
        print(f"WARNING: could not read the prompt-caching setting ({type(e).__name__}: {e}).")
        return
    if disabled:
        print("Prompt caching is disabled for this project.")
    else:
        print("WARNING: prompt caching is ENABLED for this project. That is fine for the "
              "synthetic sample, but turn it off before sending real patient data:\n"
              f"  curl -X PATCH -H \"Authorization: Bearer $(gcloud auth print-access-token)\" "
              f"-H \"Content-Type: application/json\" {url} -d '{{\"disableCache\": true}}'")


def build_generation_config(generation):
    """spec["generation"] -> Gemini config. thinking_budget moves into thinking_config."""
    config = dict(generation)
    if "thinking_budget" in config:
        config["thinking_config"] = {"thinking_budget": config.pop("thinking_budget")}
    return config


def generate(client, model, conversations, gen_config, mode="stream", max_workers=8,
             gcs_bucket=None, gcs_prefix=None):
    """Run every conversation through Gemini.

    Conversations use OpenAI-style roles (system / user / assistant). Returns one
    {"text", "error", "compute"} dict per conversation, in input order.
    """
    if mode == "batch":
        if not gcs_bucket:
            raise SystemExit("--gcs_bucket is required with --mode batch.")
        return _generate_batch(client, model, conversations, gen_config, gcs_bucket, gcs_prefix)
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(conversations)))) as pool:
        jobs = pool.map(lambda c: _stream_one(client, model, c, gen_config), conversations)
        return list(tqdm(jobs, total=len(conversations), desc="Gemini"))


def _to_gemini(messages):
    """Split messages into (system_instruction, contents). Gemini calls the assistant 'model'."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    return system, contents


def _result(text, usage, finish_reason, seconds):
    """usage is an SDK object (stream) or a dict from batch output."""
    def get(name):
        if isinstance(usage, dict):
            return usage.get(_camel_case(name), usage.get(name))
        return getattr(usage, name, None)

    return {
        "text": text,
        "error": None,
        "compute": {
            "input_tokens": get("prompt_token_count"),
            "output_tokens": get("candidates_token_count"),
            "thoughts_token_count": get("thoughts_token_count"),
            "cached_content_token_count": get("cached_content_token_count"),
            "generation_time_sec": seconds,
            "truncated": "MAX_TOKENS" in str(finish_reason),
        },
    }


def _error(message):
    return {"text": "", "error": message, "compute": {}}


def _stream_one(client, model, messages, gen_config):
    """One real-time request. Streaming avoids timeouts on long outputs."""
    system, contents = _to_gemini(messages)
    config = dict(gen_config)
    if system:
        config["system_instruction"] = system
    start = time.time()
    try:
        parts, usage, finish = [], None, None
        for chunk in client.models.generate_content_stream(model=model, contents=contents,
                                                           config=config):
            parts.append(chunk.text or "")
            usage = chunk.usage_metadata or usage
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish = chunk.candidates[0].finish_reason
        return _result("".join(parts), usage, finish, round(time.time() - start, 3))
    except Exception as e:
        print(f"Request failed: {type(e).__name__}: {e}")
        return _error(f"{type(e).__name__}: {e}")


def _camel_case(name):
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)


def _camel(obj):
    """snake_case keys -> camelCase, as the batch REST format expects."""
    if isinstance(obj, dict):
        return {_camel_case(k): _camel(v) for k, v in obj.items()}
    return obj


def _batch_request(messages, gen_config):
    """One JSONL request body for a Vertex AI batch job."""
    system, contents = _to_gemini(messages)
    request = {"contents": contents}
    if system:
        request["system_instruction"] = {"parts": [{"text": system}]}
    if gen_config:
        request["generation_config"] = _camel(gen_config)
    return request


def _batch_jsonl(conversations, gen_config):
    """JSONL payload; each line's "key" lets results be matched back to rows."""
    return "".join(json.dumps({"key": f"row-{i:06d}", "request": _batch_request(m, gen_config)},
                              ensure_ascii=False) + "\n"
                   for i, m in enumerate(conversations))


def _generate_batch(client, model, conversations, gen_config, bucket_name, prefix,
                    poll_seconds=30):
    """Upload JSONL to Cloud Storage, run a Vertex AI batch job, read the results back."""
    from google.cloud import storage
    from google.genai.types import CreateBatchJobConfig

    n = len(conversations)
    prefix = prefix.strip("/")
    bucket = storage.Client().bucket(bucket_name)
    bucket.blob(f"{prefix}/input.jsonl").upload_from_string(
        _batch_jsonl(conversations, gen_config), content_type="application/jsonl")

    job = client.batches.create(
        model=model, src=f"gs://{bucket_name}/{prefix}/input.jsonl",
        config=CreateBatchJobConfig(dest=f"gs://{bucket_name}/{prefix}/output"))
    print(f"Submitted batch job {job.name} ({n} requests). Polling every {poll_seconds}s...")
    state = job.state.name if job.state else ""
    while state not in BATCH_DONE_STATES:
        time.sleep(poll_seconds)
        job = client.batches.get(name=job.name)
        state = job.state.name if job.state else ""
    print(f"Batch job finished: {state}")
    if state != "JOB_STATE_SUCCEEDED":
        print(f"Keeping gs://{bucket_name}/{prefix}/ for debugging; delete it when done.")
        return [_error(f"batch job {state}: {job.error}") for _ in range(n)]

    # Output order is not guaranteed; match each line back by its key.
    index = {f"row-{i:06d}": i for i in range(n)}
    results = [None] * n
    for blob in bucket.client.list_blobs(bucket_name, prefix=f"{prefix}/output"):
        if not blob.name.endswith(".jsonl"):
            continue
        for line in blob.download_as_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            i = index.get(record.get("key"))
            if i is None:
                continue
            response = record.get("response")
            if not response:
                status = record.get("status") or record.get("error")
                results[i] = _error(f"batch item error: {status}")
                continue
            first = (response.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (first.get("content") or {}).get("parts", []))
            usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
            finish = first.get("finishReason") or first.get("finish_reason")
            results[i] = _result(text, usage, finish, None)

    missing = [i for i in range(n) if results[i] is None]
    for i in missing:
        results[i] = _error("no batch result for this row")
    if missing:
        print(f"WARNING: {len(missing)} row(s) had no result; keeping gs://{bucket_name}/{prefix}/ "
              "for debugging.")
    else:
        # Staged files contain patient text; delete them once results are saved.
        for blob in list(bucket.client.list_blobs(bucket_name, prefix=f"{prefix}/")):
            blob.delete()
    return results
