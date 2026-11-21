"""Strata's tokenizer (tools/strata_tokenizer.py, the `inkling2` rule) against llama.cpp's on an Inkling 2 GGUF.

    python terminal/tools/inkling/tokenizer_check.py --gguf Inkling-2-Mini-UD-IQ1_S.gguf --llama-tokenize llama.cpp/build/bin/llama-tokenize

The corpus hits what a pre-tokenizer gets wrong without a visible error: case runs and the English contractions
(the GPT-4o split keeps "HTTPServer" as "HTTP" + "Server"), digit runs of at most three, punctuation followed by
newlines and slashes, CJK and Arabic, emoji and combining marks, whitespace runs, and the chat template's control
tokens (parsed as specials). Every string must give the same ids and decode back to itself.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import strata_tokenizer as ST  # noqa: E402

CORPUS = [
    "Hello, world!",
    "The capital of France is Paris. The capital of Germany is",
    "HTTPServer and XMLHttpRequest, getHTTPResponseCode()",
    "I'm sure they'll say we've done it; you'd know, isn't it? DON'T SHOUT, I'M HERE",
    "1234567890 3.14159 1,000,000 2026-11-21 0x7fff",
    "a/b/c path/to/file.txt https://example.com/a?b=c&d=e",
    "line one\nline two\n\n\nline five\r\n\tindented",
    "    leading spaces and trailing   ",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n",
    "你好，世界！今天天气很好。",
    "مرحبا بالعالم",
    "こんにちは世界 カタカナ",
    "\U0001f600\U0001f680 \U0001f1fa\U0001f1f8 \U0001f44d\U0001f3fd",
    "été café नमस्ते สวัสดี",
    "!!!??? ... --- *** ### ```json\n{\"a\": [1, 2]}\n```",
    "<|message_user|><|content_text|>What is 2+2?<|end_message|><|message_model|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
    "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|><|content_model_end_sampling|>",
]


def llama_ids(exe: str, gguf: str, text: str) -> list[int]:
    out = subprocess.run([exe, "-m", gguf, "-p", text, "--ids", "--log-disable"], capture_output=True, text=True, timeout=300)
    line = [x for x in out.stdout.strip().splitlines() if x.startswith("[")][-1]
    return [int(x) for x in line.strip("[]").replace(" ", "").split(",") if x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-tokenize", required=True)
    a = ap.parse_args()
    tk = ST.Tokenizer.from_gguf(a.gguf)
    print(f"pre {tk.pre!r}, vocab {len(tk.tokens)}")
    bad = 0
    for text in CORPUS:
        mine = tk.encode(text, parse_special=True)
        ref = llama_ids(a.llama_tokenize, a.gguf, text)
        back = tk.decode(mine)
        ok = mine == ref and back == text
        bad += not ok
        print(("ok   " if ok else "FAIL ") + repr(text[:60]) + ("" if ok else f"\n     strata {mine[:24]}\n     llama  {ref[:24]}"))
    print(f"{len(CORPUS) - bad} of {len(CORPUS)} agree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
