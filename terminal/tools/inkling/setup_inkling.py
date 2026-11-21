"""Set up Inkling 2 Mini with the strata-inkling engine from a source checkout, and optionally start the server.

    python terminal/tools/inkling/setup_inkling.py                       # the size for this computer's RAM, then a plan
    python terminal/tools/inkling/setup_inkling.py --model UD-Q4_K_XL --yes --start

Steps: the size (strata_control/inkling_catalog.py: the file, its pinned revision, size and SHA-256), the RAM plan (every
expert in RAM, or the dense weights in RAM and the experts streamed from the SSD through a cache), the engine (built from
engine/ with CMake unless --engine names one), the download (resumable, checked), the tokenizer from the GGUF, and the run
config for terminal/serve/server.py. Nothing is downloaded or built with --plan.

This is the interim path until strata-inkling ships in the release packages; docs/INKLING.md says what is and is not done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]          # the checkout
TERMINAL = ROOT / "terminal"
sys.path[:0] = [str(TERMINAL), str(TERMINAL / "tools")]

from strata_control import inkling_catalog as C  # noqa: E402


def available_ram_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / 2 ** 30 * 0.7
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("pt", ctypes.c_ulonglong), ("pa", ctypes.c_ulonglong),
                        ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong), ("ve", ctypes.c_ulonglong)]
        m = MS()
        m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 2 ** 30
    return 8.0


def say(s: str = ""):
    print(s, flush=True)


def download(url: str, dest: pathlib.Path, size: int, sha: str):
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and dest.stat().st_size == size:
        say(f"  have {dest.name}; checking it ...")
    else:
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as out:
            done, last = have, -1
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                pct = int(100 * done / size)
                if pct != last and pct % 5 == 0:
                    say(f"  {dest.name}: {done / 1e9:.1f} of {size / 1e9:.1f} GB")
                    last = pct
        if part.stat().st_size != size:
            raise SystemExit(f"the download of {dest.name} stopped at {part.stat().st_size} of {size} bytes: run again to resume")
        part.replace(dest)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        raise SystemExit(f"{dest.name} does not match its SHA-256 ({h.hexdigest()}): delete it and run again")
    say(f"  {dest.name}: SHA-256 checked")


def build_engine() -> pathlib.Path:
    build = ROOT / "engine" / "build" / "inkling"
    exe = build / ("strata-inkling.exe" if os.name == "nt" else "strata-inkling")
    if exe.exists():
        return exe
    if not shutil.which("cmake"):
        raise SystemExit("CMake is needed to build strata-inkling (or pass --engine with a built one)")
    say("  building strata-inkling (a few minutes; the CPU engine needs no GPU toolchain) ...")
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    subprocess.run(["cmake", "-S", str(ROOT / "engine"), "-B", str(build), *gen, "-DCMAKE_BUILD_TYPE=Release",
                    "-DSTRATA_BUILD_TESTS=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "strata-inkling", "-j"], check=True)
    if not exe.exists():
        alt = build / "Release" / exe.name
        if alt.exists():
            return alt
        raise SystemExit("the build did not produce strata-inkling")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(C.SIZES), help="the size (default: by this computer's RAM)")
    ap.add_argument("--data-dir", default=str(pathlib.Path.home() / "strata-inkling"))
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--ram-budget", type=float, help="GiB the engine may use (default: what is free less a margin)")
    ap.add_argument("--engine", help="a built strata-inkling (default: build it from engine/)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--plan", action="store_true", help="say what would happen and stop")
    ap.add_argument("--yes", action="store_true", help="no questions")
    ap.add_argument("--start", action="store_true", help="start the server when done")
    a = ap.parse_args()

    ram = a.ram_budget if a.ram_budget else available_ram_gib()
    size = a.model or C.recommend(ram, a.context)
    file, nbytes, sha, dense, experts, about = C.SIZES[size]
    p = C.plan(size, ram, a.context)
    say(f"{C.TITLE} - {C.BY}: {C.ABOUT}")
    say(f"  license: {C.LICENSE}")
    say(f"  size: {size} ({about}); download {nbytes / 1e9:.1f} GB")
    say(f"  RAM: {ram:.1f} GiB {'given' if a.ram_budget else 'free'}; dense weights {dense:.2f} GiB, experts {experts:.2f} GiB, "
        f"attention caches {a.context * C.KV_BYTES_PER_TOKEN / 2 ** 30 + C.RING_GIB:.2f} GiB at {a.context} tokens")
    if p["mode"] == "ram":
        say("  plan: every expert in RAM")
    elif p["mode"] == "stream":
        say(f"  plan: {p['expert_cache_gib']:.1f} GiB of experts in RAM ({p['share'] * 100:.0f}%), the rest read from the SSD as "
            "needed (an NVMe SSD keeps this fast; a hard disk does not)")
    else:
        say(f"  plan: too little RAM for {size} at {a.context} tokens; choose a smaller --context or size")
        if not a.yes:
            return 2
    data = pathlib.Path(a.data_dir).expanduser()
    if a.plan:
        say(f"  would download to {data / file}, build the engine, and write {data / 'strata-inkling.json'}")
        return 0
    if not a.yes and input("  Go on? [Y/n] ").strip().lower() not in ("", "y", "yes"):
        return 1
    data.mkdir(parents=True, exist_ok=True)
    exe = pathlib.Path(a.engine).resolve() if a.engine else build_engine()
    url = f"https://huggingface.co/{C.REPO}/resolve/{C.REVISION}/{file}"
    download(url, data / file, nbytes, sha)
    import strata_tokenizer as ST
    cfg_t = ST.extract(data / file, data)
    say(f"  tokenizer: vocab {cfg_t['vocab_size']}, pre {cfg_t['pre']}")
    args = ["--model", str(data / file), "--ctx", str(a.context)]
    if a.ram_budget:
        args += ["--ram-budget", str(a.ram_budget)]
    cfg = {"arch": C.ARCH, "model_name": f"{C.NAME}-{size.lower()}", "exe": str(exe), "args": args,
           "tokenizer": str(data / "tokenizer"), "cwd": str(data), "log": str(data / "engine.log"), "port": a.port}
    conf = data / "strata-inkling.json"
    conf.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    cmd = [sys.executable, str(TERMINAL / "serve" / "server.py"), "--engine", "strata", "--config", str(conf), "--port", str(a.port)]
    say(f"  config: {conf}")
    say("  start the server with:  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if a.start:
        return subprocess.call(cmd, cwd=str(TERMINAL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
