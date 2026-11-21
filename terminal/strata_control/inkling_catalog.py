"""strata_control/inkling_catalog.py - Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) as Unsloth publishes it in GGUF
(unsloth/Inkling-2-Mini-GGUF), for the strata-inkling engine (engine/src/inkling, docs/INKLING.md).

Each size lists the file (pinned revision, size, SHA-256 from the Hub) and what the engine keeps where: the dense weights
always in RAM, the experts in RAM when they fit and otherwise streamed from the SSD through a RAM cache. The GiB figures
are the tensors in each file's header (dense, experts), measured on 2026-11-21.
"""
from __future__ import annotations

REPO = "unsloth/Inkling-2-Mini-GGUF"
REVISION = "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f"
TITLE = "Inkling 2 Mini"
BY = "Thinking Machines Lab; GGUFs by Unsloth"
ABOUT = ("26.4B mixture of experts, 3.4B active (128 experts, top-8), middle 12 layers run three times per token; "
         "reads text (pictures are not supported by this engine yet)")
LICENSE = "Apache 2.0: https://huggingface.co/thinkingmachines/Inkling-2-Mini"
ARCH = "inkling2"
NAME = "inkling-2-mini"

# size -> file, bytes, sha256, dense GiB, experts GiB, one line for the menu
SIZES = {
    "UD-IQ1_S": ("Inkling-2-Mini-UD-IQ1_S.gguf", 7764291776,
                 "c09c028b050c108624682baeb28ac6e0cdb936af021ac637c8c3609f923ea4b8", 1.14, 6.08,
                 "1.6-bit experts, the smallest"),
    "UD-IQ1_M": ("Inkling-2-Mini-UD-IQ1_M.gguf", 8024441024,
                 "544e0b2412ba2392272b0510b0c8f53a1ab872b73c2ba4f7488de0d5d61467d9", 1.12, 6.34, "1.75-bit experts"),
    "UD-IQ2_XXS": ("Inkling-2-Mini-UD-IQ2_XXS.gguf", 8486236352,
                   "4fe958f73a667d8082f911a4347e52bc34696a7816cf3d5b9f7c8a81d5c5ceaa", 1.14, 6.75, "2-bit experts"),
    "UD-Q2_K_XL": ("Inkling-2-Mini-UD-Q2_K_XL.gguf", 9138715328,
                   "20292e8142b69535ca88a5d04439d0fdeb4dc698d44c002dca706dcf92901441", 1.15, 7.35, "2-bit experts, more NVFP4"),
    "UD-IQ3_XXS": ("Inkling-2-Mini-UD-IQ3_XXS.gguf", 11663309504,
                   "bd7ef0865adf119b5183e45e741c629bafa649769b33b5ebf689bb4bcc4690d6", 1.26, 9.59, "3-bit experts"),
    "UD-Q3_K_XL": ("Inkling-2-Mini-UD-Q3_K_XL.gguf", 13631778016,
                   "f72f396d1b479b14d76346df80baa449c8b1f399653bff38fa4764b359a40048", 1.32, 11.36, "3-bit experts, mostly NVFP4"),
    "UD-IQ4_XS": ("Inkling-2-Mini-UD-IQ4_XS.gguf", 15733755232,
                  "a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1", 1.57, 13.07, "4-bit, the original NVFP4 experts"),
    "UD-Q4_K_M": ("Inkling-2-Mini-UD-Q4_K_M.gguf", 16610543328,
                  "f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1", 1.69, 13.76, "4-bit, the original NVFP4 experts"),
    "UD-Q4_K_XL": ("Inkling-2-Mini-UD-Q4_K_XL.gguf", 17629759456,
                   "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15", 1.69, 14.71,
                   "the checkpoint's own NVFP4 experts, the recommended size"),
    "UD-Q8_K_XL": ("Inkling-2-Mini-UD-Q8_K_XL.gguf", 22867451616,
                   "dfb04bf37bee457f98800989dedf93ec442ede8636dbea9433992f1e527e9a41", 3.01, 18.27, "the closest to the checkpoint"),
}
DEFAULT = "UD-Q4_K_XL"

# The attention caches per context token: 8 of the 48 layer applications attend to every position, K and V of 8 heads x 128
# in f16; the other 40 keep a ring of 1,024 + one prompt chunk (fixed, about 0.2 GiB).
KV_BYTES_PER_TOKEN = 8 * 2 * 8 * 128 * 2
RING_GIB = 40 * (1024 + 256) * 2 * 8 * 128 * 2 / 2 ** 30
# below this much for experts the cache thrashes (measured: a 1.5 GiB cache of UD-Q4_K_XL still answers 61% of lookups from RAM)
MIN_EXPERT_CACHE_GIB = 1.0
OS_MARGIN_GIB = 1.0


def need_gib(size: str, context: int, expert_cache_gib: float | None = None) -> float:
    """The RAM the engine takes for `size` at `context` tokens: dense + caches + buffers + the expert cache (all experts
    when `expert_cache_gib` is None)."""
    _, _, _, dense, experts, _ = SIZES[size]
    cache = experts if expert_cache_gib is None else min(experts, expert_cache_gib)
    return dense + context * KV_BYTES_PER_TOKEN / 2 ** 30 + RING_GIB + 0.3 + cache


def plan(size: str, ram_gib: float, context: int) -> dict:
    """How `size` runs with `ram_gib` of free RAM: "ram" (every expert in RAM), "stream" (some from the SSD, with the
    expert cache it gets) or "too_small"."""
    _, _, _, dense, experts, _ = SIZES[size]
    room = ram_gib - OS_MARGIN_GIB - need_gib(size, context, 0.0)
    if room >= experts:
        return {"mode": "ram", "expert_cache_gib": experts, "share": 1.0}
    if room >= MIN_EXPERT_CACHE_GIB:
        return {"mode": "stream", "expert_cache_gib": round(room, 2), "share": round(room / experts, 2)}
    return {"mode": "too_small", "expert_cache_gib": max(0.0, round(room, 2)), "share": 0.0}


def recommend(ram_gib: float, context: int) -> str:
    """The size to offer first.  Streaming is what this engine is for, so the recommended UD-Q4_K_XL is offered whenever
    its expert cache gets at least a tenth of its experts (measured on UD-Q4_K_XL, docs/INKLING.md: 10% of the experts in
    RAM answered 61% of lookups from RAM, 20% 81%, 41% 90%); below that, the largest size whose experts all fit, else the
    smallest."""
    p = plan(DEFAULT, ram_gib, context)
    if p["mode"] == "ram" or p["share"] >= 0.10:
        return DEFAULT
    for size in reversed(list(SIZES)):
        if plan(size, ram_gib, context)["mode"] == "ram":
            return size
    return "UD-IQ1_S"
