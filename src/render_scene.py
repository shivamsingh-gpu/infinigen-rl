"""Render a SceneConfig with Infinigen, or with a stub for smoke-testing.

The Infinigen path shells out to its `manage_jobs` / `generate.py` entrypoint
via subprocess so this file has no hard dep on Blender's Python. If your local
Infinigen install exposes a different CLI, edit `_INFINIGEN_CMD`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from config_space import SceneConfig

# Infinigen root: sibling `../infinigen` by default (matches this scaffold's
# layout). Override with INFINIGEN_ROOT env var. Resolved to absolute so that
# subprocess `cwd=` works no matter where python is invoked from.
INFINIGEN_ROOT = Path(os.environ.get("INFINIGEN_ROOT", Path(__file__).resolve().parents[2])).resolve()


def render_stub(cfg: SceneConfig, out_dir: Path, size: int = 512) -> Path:
    """Cheap placeholder render — a colored card labeled with the config.
    Lets you exercise the loop and reward without paying Blender's warmup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "forest": (46, 92, 54),
        "desert": (214, 173, 96),
        "mountain": (120, 124, 130),
        "coast": (86, 141, 168),
        "arctic": (220, 230, 236),
    }
    img = Image.new("RGB", (size, size), palette[cfg.biome])
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"{cfg.biome} / {cfg.time_of_day}\n{cfg.weather}", fill=(255, 255, 255))
    out = out_dir / "render.png"
    img.save(out)
    return out


# Biomes we expose to the LLM → the gin file Infinigen ships in
# src/infinigen_examples/configs_nature/scene_types/*.gin
BIOME_TO_GIN = {
    "forest": "forest.gin",
    "desert": "desert.gin",
    "mountain": "mountain.gin",
    "coast": "coast.gin",
    "arctic": "arctic.gin",
}

INFINIGEN_PY = INFINIGEN_ROOT / ".venv" / "bin" / "python"


FAST_MODE = os.environ.get("RL_FAST_RENDER", "1") != "0"


def _gpu_count() -> int:
    """How many GPUs this job can see. Prefer the Slurm/CUDA-visible list."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        return max(1, len([x for x in cvd.split(",") if x.strip() != ""]))
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        n = out.stdout.count("GPU ")
        return max(1, n)
    except Exception:
        return 1


def _affinity(slot: int):
    """Map a render worker `slot` to one dedicated GPU + a disjoint CPU set.

    The trainer launches RENDER_PARALLELISM Blender subprocesses at once. Left
    unpinned, each one grabs *all* cores (Blender/Cycles default) and, in the
    render stage, *all* GPUs (configure_cycles_devices enables every device of
    the preferred type) — so 8 processes oversubscribe 96 cores ~8x (coarse
    stage thrashes) and pile onto the same GPUs. Pinning one GPU + a core slice
    per worker gives clean N-way parallelism: coarse stops thrashing, and each
    scene renders on its own A100.

    Returns (cuda_visible_devices: str, taskset_cpulist: str|None, n_threads: int).
    """
    n_gpu = _gpu_count()
    # GPU 0 hosts the training policy (train_grpo runs on cuda:0). Keep render
    # workers off it when we have spare GPUs, so a render's VRAM peak can never
    # collide with the policy's forward/backward on GPU 0. With a single GPU we
    # have no choice and share it.
    gpu = (1 + slot % (n_gpu - 1)) if n_gpu > 1 else 0
    # Pin to a disjoint slice of the cores THIS process is actually allowed to
    # use. os.cpu_count() reports the machine's physical cores (96 on a p4d),
    # NOT the cgroup/cpuset allocation -- so under a partial `--cpus-per-task`
    # (e.g. 32) it mapped high slots to cores 48-95 that were not allocated,
    # `taskset` failed with "Invalid argument", the coarse subprocess never
    # launched, and the caller silently reused a stale frame. sched_getaffinity
    # gives the real allowed set (and may be non-contiguous), so slice from it.
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        allowed = list(range(os.cpu_count() or 8))
    ncpu = len(allowed) or 1
    per = max(1, ncpu // n_gpu)          # cores per worker slot
    start = (slot % n_gpu) * per
    cores = allowed[start:start + per] or allowed[:1]
    cpulist = ",".join(str(c) for c in cores)  # taskset -c accepts a comma list
    return str(gpu), cpulist, len(cores)


def _run_stage(task: str, cfg: SceneConfig, gin: str, in_dir: Path | None,
               out_dir: Path, slot: int) -> None:
    gins = [gin, "simple.gin"]
    if FAST_MODE:
        gins.append("rl_fast.gin")

    gpu, cpulist, n_threads = _affinity(slot)

    # Infinigen's parse_seed reads --seed as HEXADECIMAL (it explicitly refuses
    # decimal to avoid hex/dec ambiguity), then np.random.seed rejects anything
    # >= 2**32. So hex-encode a uint32 here: Infinigen decodes int(hex, 16) back
    # to the same value, guaranteed in [0, 2**32-1].
    seed_hex = format(int(cfg.seed) % (2 ** 32), "x")
    cmd = [
        str(INFINIGEN_PY), "-m", "infinigen_examples.generate_nature",
        "--seed", seed_hex,
        "--task", *task.split(),
        "-g", *gins,
        "--output_folder", str(out_dir),
    ]
    if in_dir is not None:
        cmd += ["--input_folder", str(in_dir)]

    # Confine this worker to one GPU + its core slice.
    if cpulist and shutil.which("taskset"):
        cmd = ["taskset", "-c", cpulist] + cmd

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # Tame the numpy/scipy/landlab thread pools that dominate the coarse
    # (erosion/snowfall) stage so parallel workers don't fight for cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS"):
        env[var] = str(n_threads)

    subprocess.run(cmd, cwd=INFINIGEN_ROOT, check=True, env=env)


def render_infinigen(cfg: SceneConfig, out_dir: Path, slot: int = 0) -> Path:
    """Run the three-stage Infinigen pipeline for one scene.

    Coarse → populate/fine_terrain → render. `slot` pins this worker to one
    GPU + a disjoint CPU set (see `_affinity`). The render stage uses the GPU
    via Infinigen's default configure_cycles_devices(use_gpu=True) (OptiX).
    Returns the path to the final RGB Image PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gin = BIOME_TO_GIN.get(cfg.biome, "desert.gin")
    coarse = out_dir / "coarse"
    fine = out_dir / "fine"
    frames = out_dir / "frames"

    _run_stage("coarse", cfg, gin, None, coarse, slot)
    _run_stage("populate fine_terrain", cfg, gin, coarse, fine, slot)
    _run_stage("render", cfg, gin, fine, frames, slot)

    rgb = frames / "Image" / "camera_0"
    pngs = sorted(rgb.glob("Image_*.png"))
    if not pngs:
        raise RuntimeError(f"no RGB frame under {rgb}")
    return pngs[0]


def render(cfg: SceneConfig, out_dir: Path, stub: bool = False, slot: int = 0) -> Path:
    return render_stub(cfg, out_dir) if stub else render_infinigen(cfg, out_dir, slot)
