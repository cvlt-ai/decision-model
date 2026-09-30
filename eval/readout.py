"""Logit-readout backend for the S1 LoRA checkpoint (M2 line).

This is the Open-Jev / JevLite recipe (arXiv 2609.23959): a small causal LM, LoRA-tuned,
whose decision is read by scoring the model's answer-key token logits — NOT by sampling
generated text. No decode, no parse, nothing to hallucinate.

Readout = conditional log-probability of each option key given the rendered prompt:

    score(key) = sum_t  log_softmax( logits[ prompt_len + t - 1 ] )[ key_token_t ]

- single-token keys (A..J, true, false, 0..3): one prompt forward, read the last
  position's logit per key — fast path
- multi-token keys (banking77 intents, go_emotions labels): teacher-force each key,
  sum its per-token logprobs — exact path

Probabilities are a softmax over the option keys (temperature defaults to 1.0; the
paper's mandatory post-hoc temperature scaling is a calibration-phase concern, not a
readout-shape concern). Answers are returned Jev-shaped so run_eval's grader consumes
them identically to Laya.

Usage:
    python -m eval.run_eval --model readout \
        --ckpt checkpoints/s1-v1 --base Qwen/Qwen3.5-4B --device cuda:1 \
        --limit 300 --out results/s1v1_300.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# single source of truth for prompt shape: train (s1.mixture) and readout must
# render identically or eval gains are meaningless. import, don't retype.
from s1.mixture import render as _mixture_render

render_prompt = _mixture_render


class LogitReadout:
    def __init__(self, ckpt: str, base: str, device: str = "cuda:1",
                 temperature: float = 1.0, max_len: int = 1024,
                 dtype: torch.dtype = torch.bfloat16, key_batch: int = 16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.temperature = temperature
        self.max_len = max_len
        self.key_batch = key_batch  # multi-token keys scored in chunks of this size
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # adapter checkpoints carry adapter_config.json; full-parameter checkpoints
        # are plain HF dirs — load them directly (and from `ckpt`, not `base`).
        ckpt_dir = Path(ckpt)
        full_ckpt = ckpt_dir.is_dir() and not (ckpt_dir / "adapter_config.json").exists() \
            and str(ckpt_dir) != str(base)
        if full_ckpt:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, dtype=dtype, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base, dtype=dtype, trust_remote_code=True
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ckpt)
        model.eval()
        model.to(device)
        self.model = model

    # ---- readout core ---------------------------------------------------

    def _key_ids(self, key: str) -> list[int]:
        return self.tok(key, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _score_keys(self, prompt: str, options: list[tuple[str, str]]) -> dict[str, float]:
        """Return {key: conditional logprob} for each option key."""
        prompt_ids = self.tok(prompt, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]
        if prompt_ids.numel() > self.max_len:
            # truncate from the left, keep the tail (options + "Answer:")
            prompt_ids = prompt_ids[-self.max_len :]

        # fast path: every key is a single token -> one forward, last-position logits
        key_tokens = {k: self._key_ids(k) for k, _ in options}
        if all(len(v) == 1 for v in key_tokens.values()):
            inp = prompt_ids.unsqueeze(0).to(self.device)  # 2D: 1D breaks RoPE
            logits = self.model(input_ids=inp).logits[0, -1]
            logp = nn.functional.log_softmax(logits / self.temperature, dim=-1)
            return {k: float(logp[tok_ids[0]]) for k, tok_ids in key_tokens.items()}

        # exact path: teacher-force each key, but score them in chunks of
        # self.key_batch (a full 77-key batch OOMs a 48 GB GPU at 1k context).
        # Prompt is fixed-length across keys, so per-chunk scoring is exact.
        out = {}
        pl = prompt_ids.numel()
        keys = [k for k, _ in options]
        for c0 in range(0, len(keys), self.key_batch):
            chunk = keys[c0 : c0 + self.key_batch]
            seqs = [torch.cat([prompt_ids, torch.tensor(key_tokens[k], dtype=prompt_ids.dtype)])
                    for k in chunk]
            L = max(s.numel() for s in seqs)
            pad_id = self.tok.pad_token_id
            inp = torch.full((len(chunk), L), pad_id, dtype=prompt_ids.dtype)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, s in enumerate(seqs):
                inp[i, : s.numel()] = s
                mask[i, : s.numel()] = 1
            inp, mask = inp.to(self.device), mask.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=inp, attention_mask=mask).logits
                logp = nn.functional.log_softmax(logits / self.temperature, dim=-1)
            for i, k in enumerate(chunk):
                s = 0.0
                for j, tok in enumerate(key_tokens[k]):
                    pos = pl - 1 + j  # logits at pos predict token at pos+1
                    s += float(logp[i, pos, tok])
                out[k] = s
        return out

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        m = max(scores.values())
        exps = {k: torch.exp(torch.tensor(v - m)).item() for k, v in scores.items()}
        z = sum(exps.values())
        return {k: v / z for k, v in exps.items()}

    # ---- Jev-shaped answer ---------------------------------------------

    def _answer(self, qtype: str, q: dict, state) -> dict:
        crit = q.get("criteria")
        if qtype == "noul":
            opts = [("true", (crit or {}).get("true", "")), ("false", (crit or {}).get("false", ""))]
            scores = self._score_keys(render_prompt(state, q["instructions"], opts), opts)
            probs = self._softmax(scores)
            return {"type": "noul", "noul": probs["true"]}
        if qtype == "choice":
            opts = list(crit.items())
            scores = self._score_keys(render_prompt(state, q["instructions"], opts), opts)
            probs = self._softmax(scores)
            return {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
            }
        # score: ordered criteria list; keys are "0".."n"
        levels = list(range(len(crit)))
        opts = [(str(i), crit[i]) for i in levels]
        scores = self._score_keys(render_prompt(state, q["instructions"], opts), opts)
        probs = self._softmax(scores)
        score = sum(i * probs[str(i)] for i in levels)
        legend = {str(i): (crit[i] if isinstance(crit[i], str) else json.dumps(crit[i]))
                  for i in levels}
        return {
            "type": "score",
            "score": float(score),
            "legend": legend,
            "probabilities": probs,
        }

    def call(self, state, questions: dict) -> dict[str, dict]:
        return {qid: self._answer(q["type"], q, state) for qid, q in questions.items()}


def build(ckpt: str, base: str, device: str = "cuda:1",
          temperature: float = 1.0, max_len: int = 1024, key_batch: int = 16):
    return LogitReadout(ckpt, base, device, temperature, max_len,
                        key_batch=key_batch).call
