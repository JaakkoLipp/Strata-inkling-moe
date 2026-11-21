"""The Inkling engine's serve protocol against itself (no server, no GPU; needs a model file).

    python terminal/tools/inkling/serve_gate.py --engine engine/build/ink/strata-inkling --model Inkling-2-Mini-UD-IQ1_S.gguf

Checks, each against a fresh engine process where it matters:
  * a conversation that continues (prompt + reply + more) reuses the held state (RESUME) and gives the same greedy
    tokens as a fresh engine reading the whole prompt;
  * a prompt that keeps the last prompt but not the reply after it (a client that writes the reply back differently,
    e.g. without its thinking) goes back to the end of the last prompt (RESUME = that length) and gives a fresh
    engine's tokens;
  * STOP ends a request at the next token (DONE cancel), GEN 0 prefill=1 reads without generating, a seeded sample repeats;
  * an id past the vocabulary is refused with ERR and the engine keeps serving.
The engine computes every token the same way whatever the chunking, so "the same" here is token for token.
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time


class Engine:
    def __init__(self, exe: str, model: str, extra: list[str]):
        self.p = subprocess.Popen([exe, "--model", model, "--serve", *extra], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        while True:
            line = self.line(900)
            if line.startswith("READY"):
                self.ctx = int(line.split()[1])
                break
            if line.startswith("ERR"):
                raise RuntimeError(line)

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def line(self, timeout: float = 600) -> str:
        got = self.q.get(timeout=timeout)
        if got is None:
            raise RuntimeError("the engine ended")
        return got

    def send(self, text: str):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    def gen(self, ids: list[int], max_new: int, keys: str = "", stop_after: int | None = None) -> dict:
        self.send(f"GEN {max_new}{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
        out = {"tokens": [], "resume": 0, "pp": 0}
        while True:
            line = self.line()
            if line.startswith("T "):
                out["tokens"].append(int(line[2:]))
                if stop_after is not None and len(out["tokens"]) == stop_after:
                    self.send("STOP")
            elif line.startswith("RESUME "):
                out["resume"] = int(line.split()[1])
            elif line.startswith("PP "):
                out["pp"] += 1
            elif line.startswith("DONE"):
                f = line.split()
                out["finish"] = f[5]
                out["reused"] = int(f[8])
                return out
            elif line.startswith("ERR"):
                out["err"] = line
                return out

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="--experts auto --ctx 4096", help="engine options")
    a = ap.parse_args()
    extra = a.extra.split()
    fails = []

    def check(cond: bool, what: str):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fails.append(what)

    # "The capital of France is Paris. The capital of Germany is" (the model's own tokens)
    prompt = [371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884]
    more = [13, 220, 151451, 149879]
    other = [13, 149917, 151451, 149879, 165985]

    e = Engine(a.engine, a.model, extra)
    t0 = time.time()
    r1 = e.gen(prompt, 12)
    check(r1.get("finish") in ("length", "stop") and len(r1["tokens"]) >= 1, f"first request: {len(r1['tokens'])} tokens")
    conv = prompt + r1["tokens"] + more
    r2 = e.gen(conv, 8)
    check(r2["resume"] == len(prompt) + len(r1["tokens"]) - 1,
          f"continuation reuses the held state (RESUME {r2['resume']}, expected {len(prompt) + len(r1['tokens']) - 1})")
    left = conv + other   # the last prompt, then not the reply it got
    r3 = e.gen(left, 8)
    check(r3["resume"] == len(conv), f"a prompt that leaves the reply goes back to the last prompt's end (RESUME {r3['resume']}, expected {len(conv)})")
    r4 = e.gen(prompt, 12, stop_after=2)
    check(r4.get("finish") == "cancel" and 2 <= len(r4["tokens"]) <= 3,
          f"STOP ends the request at the next token ({r4.get('finish')}, {len(r4['tokens'])} tokens)")
    r5 = e.gen(prompt + more, 0, "prefill=1")
    check(r5.get("finish") == "length" and not r5["tokens"], "GEN 0 prefill=1 reads the prompt and makes nothing")
    s1 = e.gen(prompt, 6, "temperature=1.0 seed=7")
    s2 = e.gen(prompt, 6, "temperature=1.0 seed=7")
    check(s1["tokens"] == s2["tokens"], f"a seeded sample repeats ({s1['tokens']})")
    bad = e.gen([1, 2, 10 ** 7], 2)
    check("err" in bad, "an id past the vocabulary is refused")
    again = e.gen(prompt, 4)
    check(again["tokens"] == r1["tokens"][:4], "the engine keeps serving after an ERR, with the same greedy tokens")
    e.close()

    f = Engine(a.engine, a.model, extra)
    f2 = f.gen(conv, 8)
    check(f2["tokens"] == r2["tokens"], "the continued conversation = a fresh engine on the whole prompt")
    f.close()
    g = Engine(a.engine, a.model, extra)
    g3 = g.gen(left, 8)
    check(g3["tokens"] == r3["tokens"], "after the rollback = a fresh engine on that prompt")
    g.close()

    print(f"{len(fails)} failed, {time.time() - t0:.0f} s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
