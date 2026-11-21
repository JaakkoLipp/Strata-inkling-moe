"""strata_control/inkling_catalog.py: the Inkling 2 Mini sizes and how much of each fits in a given RAM."""
from __future__ import annotations

import re

from strata_control import inkling_catalog as C


def test_every_size_is_pinned():
    for size, (file, nbytes, sha, dense, experts, about) in C.SIZES.items():
        assert file == f"Inkling-2-Mini-{size}.gguf"
        assert nbytes > 7e9 and re.fullmatch(r"[0-9a-f]{64}", sha)
        assert 1.0 < dense < 3.5 and experts > 5 and about
        assert dense + experts < nbytes / 2 ** 30 + 0.05      # the header's tensors fit in the file
    assert C.DEFAULT in C.SIZES and re.fullmatch(r"[0-9a-f]{40}", C.REVISION)


def test_plan_modes():
    assert C.plan("UD-Q4_K_XL", 32, 32768)["mode"] == "ram"
    p = C.plan("UD-Q4_K_XL", 8, 32768)
    assert p["mode"] == "stream" and 0 < p["share"] < 1 and p["expert_cache_gib"] >= C.MIN_EXPERT_CACHE_GIB
    assert C.plan("UD-IQ1_S", 3, 32768)["mode"] == "too_small"


def test_a_longer_context_leaves_less_for_experts():
    a = C.plan("UD-Q4_K_XL", 10, 8192)["expert_cache_gib"]
    b = C.plan("UD-Q4_K_XL", 10, 65536)["expert_cache_gib"]
    assert a > b


def test_recommend_streams_the_recommended_size_when_a_tenth_fits():
    assert C.recommend(32, 32768) == "UD-Q4_K_XL"
    assert C.recommend(6, 16384) == "UD-Q4_K_XL"     # 16% of its experts in RAM
    assert C.recommend(4.5, 16384) == "UD-IQ1_S"     # less than a tenth: the smallest, streamed
