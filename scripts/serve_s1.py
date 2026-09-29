#!/usr/bin/env python
"""Thin FastAPI server for the s1 decision model, speaking the Jev wire spec.

Wires together the two pieces that already exist:
  * src/s1/schema.py   — request/response validation + rendering (the wire format)
  * eval/readout.py    — LogitReadout: single-token logit scoring over option keys

Endpoints:
  GET  /healthz          -> liveness + model info
  POST /v1/answer        -> {state, questions} -> {qid: answer}  (Jev wire spec)

The readout consumes *internal* question dicts of the form
  choice: {type, instructions, criteria: {key: description}}
  score:  {type, instructions, criteria: {0: desc, 1: desc, ...}}
  noul:   {type, instructions, criteria: {true: desc, false: desc}}
so this module translates the validated DecideRequest into that shape and hands
each question to LogitReadout._answer (via .call).

Run:
  python scripts/serve_s1.py --ckpt checkpoints/s1-v6 --base Qwen/Qwen3.5-4B \
     --device cuda:0 --temperature 1.5 --max-len 1024 --key-batch 8 --port 8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Add ROOT and ROOT/src only — NOT ROOT/eval. If ROOT/eval is on the path, then
# `from datasets import ...` inside s1.mixture resolves to the local eval/datasets/
# package instead of the top-level `datasets` lib (the same shadowing the flip probe
# hit). Readout is imported as a package (eval.readout) with ROOT on the path.
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from s1.schema import parse_request, ParseError  # noqa: E402


def _internal_question(q) -> dict:
    """Translate a schema.Question into the readout's internal question dict.

    schema.Question has: qtype (alias 'type'), instructions, criteria, option_keys().
    The readout wants {type, instructions, criteria} where criteria is the
    {key: description} mapping (choice), ordered levels (score), or {true,false} (noul).
    """
    qtype = q.qtype
    instructions = q.instructions
    crit = q.criteria
    if qtype == "choice":
        # criteria may be {key: description} or {key: {description: ...}}
        out = {}
        for key in q.option_keys():
            v = (crit or {}).get(key, "")
            if isinstance(v, dict):
                v = v.get("description", "") or v.get("label", "") or ""
            out[key] = v
        return {"type": qtype, "instructions": instructions, "criteria": out}
    if qtype == "score":
        if isinstance(crit, dict):
            levels = {int(k): v for k, v in crit.items()}
        elif isinstance(crit, list):
            levels = {i: v for i, v in enumerate(crit)}
        else:
            levels = {}
        return {"type": qtype, "instructions": instructions,
                "criteria": {k: levels[k] for k in sorted(levels)}}
    if qtype == "noul":
        c = crit if isinstance(crit, dict) else {}
        return {"type": qtype, "instructions": instructions,
                "criteria": {"true": c.get("true", ""), "false": c.get("false", "")}}
    raise HTTPException(422, f"unsupported question type: {qtype}")


class AnswerBody(BaseModel):
    state: object
    questions: dict[str, object]


def build_app(ckpt: str, base: str, device: str, temperature: float,
              max_len: int, key_batch: int):
    from eval.readout import LogitReadout

    ro = LogitReadout(ckpt=ckpt, base=base, device=device, temperature=temperature,
                      max_len=max_len, key_batch=key_batch)
    model = ro.call  # bound: call(state, questions)

    app = FastAPI(title="s1 decision model", version="s1-v6")
    app.state.info = {"checkpoint": ckpt, "base": base, "device": device,
                      "temperature": temperature, "max_len": max_len}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, **app.state.info}

    @app.post("/v1/answer")
    def answer(body: AnswerBody):
        t0 = time.perf_counter()
        try:
            req = parse_request(body.model_dump())
        except ParseError as e:
            raise HTTPException(422, str(e))
        internal = {qid: _internal_question(q) for qid, q in req.questions.items()}
        state = req.state if isinstance(req.state, str) else req.state
        answers = model(state, internal)
        return {"answers": answers, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/s1-v6")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temperature", type=float, default=1.5,
                    help="release-calibrated T (see MODEL_CARD.md §5: best T=1.5)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--key-batch", type=int, default=8,
                    help="77-key families (banking77) OOM above ~16; keep 8")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn
    app = build_app(a.ckpt, a.base, a.device, a.temperature, a.max_len, a.key_batch)
    print(f"s1-v6 serving on {a.host}:{a.port} (T={a.temperature}, max_len={a.max_len})",
          file=sys.stderr)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
