#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantics_evi_score_dist.py
多卡多模型批处理（证据提取 + 计分），支持：
- 结果复用（空证据清理 + 非空判定）
- 每 GPU 独立日志文件
- LIVE 模式：父进程集中渲染进度条，避免交错显示
- 可选严格绑定核查（视频名 ↔ JSON）
- 可选串行输出（--serialize-output），完全消除交错

REF：输入目录不分主题；输出在 --ref-out-dir（平铺）。
GEN：按模型子目录；输出镜像相对路径到 --gen-out-root/<model>/...
"""

import argparse, os, sys, json, shutil, subprocess, tempfile, threading, queue, re, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ------------------ 基础 ------------------
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".flv", ".ts", ".mpg", ".mpeg", ".wmv"}

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def list_videos(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file() and is_video(root):
        return [root]
    vids = []
    for p in root.rglob("*"):
        if is_video(p):
            vids.append(p)
    return sorted(vids)

def out_path_ref(in_file: Path, ref_out_root: Path) -> Path:
    return (ref_out_root / in_file.with_suffix(".json").name).resolve()

def out_path_gen(in_file: Path, model_in_root: Path, model_out_root: Path) -> Path:
    rel = in_file.relative_to(model_in_root).with_suffix(".json")
    return (model_out_root / rel).resolve()

def _to_number(x):
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip()
        for ch in [">","<","~","+"]: s = s.replace(ch,"")
        try: return float(s)
        except: return 0.0
    return 0.0

def has_nonempty_evidence(p: Path, min_bytes: int = 64) -> bool:
    try:
        if (not p.is_file()) or p.stat().st_size < min_bytes: return False
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict): return False

        fine = obj.get("fine")
        if isinstance(fine, dict):
            if isinstance(fine.get("entities"), list) and fine["entities"]: return True
            if isinstance(fine.get("relations"), list) and fine["relations"]: return True

        fws = obj.get("fine_windows")
        if isinstance(fws, list):
            for w in fws:
                if not isinstance(w, dict): continue
                if isinstance(w.get("entities"), list) and w["entities"]: return True
                if isinstance(w.get("relations"), list) and w["relations"]: return True

        views = obj.get("views")
        if isinstance(views, dict):
            oc = views.get("objects_count")
            if isinstance(oc, dict) and any((isinstance(v,(int,float)) and v>0) for v in oc.values()): return True
            trips = views.get("triplets")
            if isinstance(trips, list) and trips: return True
            attrs = views.get("attributes")
            if isinstance(attrs, dict):
                for v in attrs.values():
                    if (isinstance(v, list) and v) or (isinstance(v, dict) and v): return True
        return False
    except:
        return False

def clean_if_empty(json_path: Path) -> None:
    try:
        if json_path.is_file() and not has_nonempty_evidence(json_path):
            json_path.unlink(missing_ok=True)
    except:
        pass

def detect_gpus(gpus_arg: str) -> List[str]:
    if gpus_arg and gpus_arg.lower() != "auto":
        return [x.strip() for x in gpus_arg.split(",") if x.strip()]
    # torch
    try:
        import torch
        n = torch.cuda.device_count()
        if n and n > 0: return [str(i) for i in range(n)]
    except: pass
    # nvidia-smi
    try:
        out = subprocess.check_output(["bash","-lc","nvidia-smi --query-gpu=index --format=csv,noheader"], text=True)
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids: return ids
    except: pass
    return []

def chunk_even(items: List, n: int) -> List[List]:
    n = max(1, n)
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def write_listfile_for_evi(tasks: List[Tuple[Path, Path]], tmp_dir: Path) -> Path:
    lf = tmp_dir / "tasks.list"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for vin, vout in tasks:
        vout.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f"{str(vin)} {str(vout)}")  # 空格分隔
    lf.write_text("\n".join(lines), encoding="utf-8")
    return lf

# ------------------ 绑定核查（可选） ------------------
def verify_bindings(tasks: List[Tuple[Path, Path]], quiet: bool = False) -> Tuple[int,int]:
    ok, bad = 0, 0
    for vin, vout in tasks:
        if not vout.is_file():
            bad += 1
            if not quiet:
                print(f"[VERIFY] MISSING: {vout}  <- {vin.name}")
            continue
        try:
            obj = json.loads(vout.read_text(encoding="utf-8"))
        except:
            obj = {}
        vbase = vin.name
        meta = obj.get("meta") if isinstance(obj, dict) else None
        matched = False
        if isinstance(meta, dict):
            vb = meta.get("video_basename") or meta.get("video_name")
            vp = meta.get("video") or meta.get("video_path")
            if isinstance(vb, str) and vb == vbase: matched = True
            elif isinstance(vp, str) and Path(vp).name == vbase: matched = True
        if not matched:
            matched = (vout.name == Path(vbase).with_suffix(".json").name)
        if matched: ok += 1
        else:
            bad += 1
            if not quiet:
                print(f"[VERIFY] NAME MISMATCH: {vout.name}  (video {vbase})")
    return ok, bad

# ------------------ 任务构建 ------------------
def build_ref_tasks(ref_video_dir: Path, ref_out_dir: Path, force: bool, limit: int = 0) -> List[Tuple[Path, Path]]:
    vids = list_videos(ref_video_dir)
    tasks = []
    for v in vids:
        o = out_path_ref(v, ref_out_dir)
        if not force: clean_if_empty(o)
        if (not force) and has_nonempty_evidence(o): continue
        tasks.append((v, o))
    if limit > 0: tasks = tasks[:limit]
    return tasks

def build_gen_tasks_for_model(model_video_dir: Path, model_out_dir: Path, force: bool, limit: int = 0) -> List[Tuple[Path, Path]]:
    vids = list_videos(model_video_dir)
    tasks = []
    for v in vids:
        o = out_path_gen(v, model_video_dir, model_out_dir)
        if not force: clean_if_empty(o)
        if (not force) and has_nonempty_evidence(o): continue
        tasks.append((v, o))
    if limit > 0: tasks = tasks[:limit]
    return tasks

# ------------------ 子进程运行 & 去交错 ------------------
_OK_RE = re.compile(r"\[OK\]\s*saved\s*->\s*(.+?\.json)\s*$")

def _spawn_worker(tag: str, evi_py: Path, model_local_path: Path, base_out_dir: Path,
                  listfile: Path, gpu_id: Optional[str], extra_opts: List[str],
                  live: bool, log_file: Optional[Path]):
    """
    返回 (Popen对象, 日志句柄或None)
    live=False -> stdout 重定向到日志文件
    live=True  -> stdout=PIPE，由父进程汇总
    """
    cmd = [sys.executable, "-u", str(evi_py),  # -u 取消缓冲
           "--batch-from", str(listfile),
           "--out-dir", str(base_out_dir),
           "--local-path", str(model_local_path),
           "--device", "cuda" if gpu_id is not None else "cpu",
           "--dtype", "bf16",
           "--decode-backend", "auto",
           "--scene-split", "--scene-thres", "0.55",
           "--scene-sample-fps", "3", "--scene-min-sec", "0.8",
           "--fps", "6", "--cap-frames", "240",
           "--resize-short", "448",
           "--window-sec", "6", "--hop-sec", "3",
           "--max-new-tokens", "512",
           "--temperature", "0.0",
           "--min-span-sec", "0.1"]
    if extra_opts: cmd += extra_opts

    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    if live:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1, text=True)
        return proc, None
    else:
        assert log_file is not None
        log_file.parent.mkdir(parents=True, exist_ok=True)
        lf = log_file.open("w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        return proc, lf

def _reader_thread(proc: subprocess.Popen, gpu_idx: int, qout: "queue.Queue[Tuple[int, Optional[str]]]"):
    """逐行读取子进程输出，送入父队列"""
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            qout.put((gpu_idx, line.rstrip("\n")))
    except Exception as e:
        qout.put((gpu_idx, f"[READER-ERR] {e}"))
    finally:
        qout.put((gpu_idx, None))  # 结束标记

def _run_buckets(tag: str, evi_py: Path, model_local_path: Path, base_out_dir: Path,
                 tasks_buckets: List[List[Tuple[Path, Path]]], gpus: List[str], extra_opts: List[str],
                 live: bool, serialize_output: bool, quiet: bool):
    """
    live=False：各 GPU 输出到独立日志；主线程打印“去哪读日志”
    live=True ：父进程集中展示 tqdm 进度条，避免交错
    serialize_output=True：逐桶串行执行（完全无交错）
    """
    tmp_root = Path(tempfile.mkdtemp(prefix=f"semantics_{tag.replace(':','_')}_"))
    log_root = tmp_root / "logs"
    procs: List[Optional[subprocess.Popen]] = []
    resources: List[Tuple[Optional[object], Optional[Path], int]] = []

    # LIVE 模式准备进度条
    if live:
        try:
            from tqdm.auto import tqdm  # noqa: F401
        except Exception:
            if not quiet:
                print("[WARN] tqdm not available; fallback to log files.")
            live = False

    # 串行时按桶逐个执行
    if serialize_output:
        for i, part in enumerate(tasks_buckets):
            if not part:
                if not quiet:
                    print(f"[{tag}] (SERIAL) bucket{i}: no tasks.")
                continue
            gpu_id = gpus[i] if i < len(gpus) else None
            lf = write_listfile_for_evi(part, tmp_root / f"gpu{i}")
            if not quiet:
                print(f"[{tag}] (SERIAL) bucket{i} -> {'GPU'+str(gpu_id) if gpu_id is not None else 'CPU'} | {len(part)} tasks")
            proc, logf = _spawn_worker(
                tag, evi_py, model_local_path, base_out_dir, lf, gpu_id,
                extra_opts, live=False, log_file=(log_root / f"gpu{i}" / f"{tag.replace(':','_')}.log")
            )
            code = proc.wait()
            if logf:
                logf.close()
            if code != 0:
                raise RuntimeError(f"[{tag}] serial bucket{i} failed with code {code}")
        if not quiet:
            print(f"[{tag}] done (serialized).")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    # 并行：根据 live 决定到日志还是管道
    for i, part in enumerate(tasks_buckets):
        gpu_id = gpus[i] if i < len(gpus) else None
        if not part:
            if not quiet:
                print(f"[{tag}] GPU{gpu_id if gpu_id is not None else 'CPU'}: no tasks.")
            procs.append(None)
            resources.append((None, None, 0))
            continue
        lf = write_listfile_for_evi(part, tmp_root / f"gpu{i}")
        if live:
            proc, _ = _spawn_worker(tag, evi_py, model_local_path, base_out_dir, lf, gpu_id, extra_opts, live=True, log_file=None)
            procs.append(proc)
            resources.append((None, None, len(part)))
        else:
            logf_path = log_root / f"gpu{i}" / f"{tag.replace(':','_')}.log"
            proc, logf = _spawn_worker(tag, evi_py, model_local_path, base_out_dir, lf, gpu_id, extra_opts, live=False, log_file=logf_path)
            if not quiet:
                print(f"[{tag}] GPU{gpu_id if gpu_id is not None else 'CPU'}: {len(part)} tasks → {lf.name} | log={logf_path}")
            procs.append(proc)
            resources.append((logf, logf_path, len(part)))

    # 非 LIVE：等子进程结束即可
    if not live:
        codes = []
        for proc, (logf, _, _) in zip(procs, resources):
            if proc is None:
                continue
            code = proc.wait()
            if logf:
                logf.close()
            codes.append(code)
        if any(c != 0 for c in codes):
            raise RuntimeError(f"[{tag}] some workers failed: {codes}")
        if not quiet:
            print(f"[{tag}] done.")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    # LIVE：集中渲染进度条 + 去交错打印关键行
    from tqdm.auto import tqdm
    totals = [r[2] for r in resources]
    bars: List[Optional["tqdm"]] = []
    for i, total in enumerate(totals):
        if total <= 0:
            bars.append(None)
            continue
        gpu_id = gpus[i] if i < len(gpus) else None
        desc = f"{tag}|GPU{gpu_id if gpu_id is not None else 'CPU'}"
        bars.append(tqdm(total=total, desc=desc, position=i, leave=True, dynamic_ncols=True))

    qout: "queue.Queue[Tuple[int, Optional[str]]]" = queue.Queue()
    threads = []
    for i, proc in enumerate(procs):
        if proc is None or proc.stdout is None:
            continue
        th = threading.Thread(target=_reader_thread, args=(proc, i, qout), daemon=True)
        th.start()
        threads.append(th)

    finished = [False] * len(procs)
    completed = [0] * len(procs)
    alive = sum(1 for p in procs if p is not None)

    while alive > 0:
        try:
            gpu_idx, line = qout.get(timeout=0.5)
        except queue.Empty:
            # 更新现存进度条(防止卡住)
            for b in bars:
                if b:
                    b.refresh()
            # 检查退出
            done_now = 0
            for i, p in enumerate(procs):
                if p is None or finished[i]:
                    continue
                if p.poll() is not None:
                    finished[i] = True
                    done_now += 1
            alive -= done_now
            continue

        if line is None:
            # 某 reader 结束
            continue

        # 匹配保存事件以推进进度
        m = _OK_RE.search(line)
        if m and bars[gpu_idx] is not None:
            bars[gpu_idx].update(1)
            completed[gpu_idx] += 1
        # 关键行可选择性打印（避免全部交错）
        if m and (not quiet):
            print(f"[{tag}|GPU{gpus[gpu_idx] if gpu_idx < len(gpus) else 'CPU'}] {m.group(0)}")

        # 检查是否有子进程退出
        if procs[gpu_idx] is not None and procs[gpu_idx].poll() is not None and not finished[gpu_idx]:
            finished[gpu_idx] = True
            alive -= 1

    # 收尾
    for b in bars:
        if b:
            b.close()
    codes = [p.wait() for p in procs if p is not None]
    if any(c != 0 for c in codes):
        raise RuntimeError(f"[{tag}] some workers failed: {codes}")
    if not quiet:
        print(f"[{tag}] done.")
    shutil.rmtree(tmp_root, ignore_errors=True)

# ------------------ 顶层流程 ------------------
def extract_evidence_dist(evi_py: Path, model_local_path: Path,
                          ref_video_dir: Optional[Path], ref_out_dir: Path,
                          gen_video_root: Path, gen_out_root: Path,
                          include_models: List[str], exclude_models: List[str],
                          gpus: List[str], force: bool, limit: int,
                          extra_evi_opts: List[str], verify: bool,
                          live: bool, serialize_output: bool, quiet: bool):
    if (not gpus) and (not quiet):
        print("[WARN] No GPUs detected, will run on CPU (very slow).")

    # REF
    if not quiet:
        print("\n=== [Stage 1/2] Extract REF evidence ===")
    ref_out_dir.mkdir(parents=True, exist_ok=True)

    # 情况 1：不提供参考视频目录 → 仅复用已有 JSON
    if ref_video_dir is None:
        existing = list(ref_out_dir.glob("*.json"))
        if existing and (not quiet):
            print(f"[Ref] ref-video-dir is None; reuse existing {len(existing)} JSON files under {ref_out_dir}")
        elif (not existing) and (not quiet):
            print(f"[Ref] ref-video-dir is None and no existing evidence found under {ref_out_dir} (nothing to extract).")
        ref_tasks: List[Tuple[Path, Path]] = []

    # 情况 2：提供了目录但目录不存在 → 打印 warning，仍然只复用已有 JSON
    elif not ref_video_dir.exists():
        if not quiet:
            print(f"[WARN] REF video dir not found: {ref_video_dir}; will only reuse existing evidence in {ref_out_dir}")
        ref_tasks = []

    # 情况 3：正常抽取参考侧证据
    else:
        ref_tasks = build_ref_tasks(ref_video_dir, ref_out_dir, force=force, limit=limit)

    if ref_tasks:
        buckets = chunk_even(ref_tasks, max(1, len(gpus)) if gpus else 1)
        _run_buckets(tag="ref", evi_py=evi_py, model_local_path=model_local_path,
                     base_out_dir=ref_out_dir, tasks_buckets=buckets, gpus=gpus or [],
                     extra_opts=extra_evi_opts, live=live, serialize_output=serialize_output,
                     quiet=quiet)
        if verify:
            ok, bad = verify_bindings(ref_tasks, quiet=quiet)
            if not quiet:
                print(f"[VERIFY/REF] ok={ok}, bad={bad}")
    else:
        if not quiet:
            print("[Skip] No REF tasks to run.")

    # GEN
    if not quiet:
        print("\n=== [Stage 2/2] Extract GEN evidence (multi-model) ===")
    if not gen_video_root.exists():
        if not quiet:
            print(f"[WARN] gen-video-root not found: {gen_video_root}")
        return

    model_dirs: List[Path] = []
    # 支持根目录直接是单模型（含视频）
    if any(is_video(p) for p in gen_video_root.iterdir() if p.is_file()):
        model_dirs = [gen_video_root]
    else:
        for d in sorted([p for p in gen_video_root.iterdir() if p.is_dir()]):
            name = d.name
            if include_models and (name not in include_models): continue
            if exclude_models and (name in exclude_models): continue
            if list_videos(d): model_dirs.append(d)

    if not model_dirs:
        if not quiet:
            print(f"[Skip] No model dirs found under: {gen_video_root}")
        return

    for md in model_dirs:
        mname = md.name
        if not quiet:
            print(f"\n--- [Model] {mname} ---")
        out_dir_m = (gen_out_root / mname).resolve()
        tasks = build_gen_tasks_for_model(md, out_dir_m, force=force, limit=limit)
        if not tasks:
            if not quiet:
                print(f"[Skip] Model {mname}: no tasks to run (all done).")
            continue
        buckets = chunk_even(tasks, max(1, len(gpus)) if gpus else 1)
        _run_buckets(tag=f"gen:{mname}", evi_py=evi_py, model_local_path=model_local_path,
                     base_out_dir=out_dir_m, tasks_buckets=buckets, gpus=gpus or [],
                     extra_opts=extra_evi_opts, live=live, serialize_output=serialize_output,
                     quiet=quiet)
        if verify:
            ok, bad = verify_bindings(tasks, quiet=quiet)
            if not quiet:
                print(f"[VERIFY/GEN:{mname}] ok={ok}, bad={bad}")

def score_softalign_per_model(batch_scoring_py: Path, yaml_path: Path,
                              ref_evi_dir: Path, gen_evi_root: Path,
                              include_models: List[str], exclude_models: List[str],
                              out_dir: Optional[Path] = None, limit: int = 0,
                              pass_through: Optional[List[str]] = None,
                              quiet: bool = False):
    if not gen_evi_root.exists():
        if not quiet:
            print(f"[WARN] gen evidence root not found: {gen_evi_root}")
        return
    model_dirs: List[Path] = []
    has_subdir = any(d.is_dir() for d in gen_evi_root.iterdir()) if gen_evi_root.exists() else False
    if has_subdir:
        for d in sorted([p for p in gen_evi_root.iterdir() if p.is_dir()]):
            name = d.name
            if include_models and (name not in include_models): continue
            if exclude_models and (name in exclude_models): continue
            if list(d.rglob("*.json")): model_dirs.append(d)
    else:
        if list(gen_evi_root.rglob("*.json")): model_dirs = [gen_evi_root]
    if not model_dirs:
        if not quiet:
            print(f"[Skip] No model evidence dirs found under: {gen_evi_root}")
        return

    for md in model_dirs:
        mname = md.name
        cmd = [sys.executable, str(batch_scoring_py),
               "--yaml", str(yaml_path),
               "--ref-dir", str(ref_evi_dir),
               "--gen-dir", str(md)]
        if out_dir:
            cmd += ["--out-dir", str(out_dir)]
        if limit and limit > 0:
            cmd += ["--limit", str(limit)]
        if pass_through:
            cmd += pass_through
        if not quiet:
            print(f"\n[Score] {mname}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"[Score] failed for model {mname} with code {proc.returncode}")

# ------------------ CLI ------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Multi-GPU Multi-Model batch runner for semantics evidence extraction + scoring (result reuse).")
    ap.add_argument("--evi-extract-py", type=Path, required=True, help="MiniCPM 证据提取脚本 evi_extract.py 路径")
    ap.add_argument("--batch-scoring-py", type=Path, required=True, help="SoftAlign 计分脚本 batch_scoring.py 路径")
    ap.add_argument("--softalign-yaml", type=Path, required=True, help="SoftAlign 配置 YAML 路径")
    ap.add_argument("--ref-video-dir", type=Path, default=None, help="参考视频根目录（不分主题）")
    ap.add_argument("--gen-video-root", type=Path, required=True, help="生成视频根目录（子目录为模型名，或直接指向单模型目录）")
    ap.add_argument("--ref-out-dir", type=Path, required=True, help="参考证据输出目录（平铺）")
    ap.add_argument("--gen-out-root", type=Path, required=True, help="生成证据输出根目录（按模型名分子目录）")
    ap.add_argument("--model-local-path", type=Path, required=True, help="MiniCPM-V-4_5 本地权重目录")

    ap.add_argument("--gpus", default="auto", help="GPU 列表，如 '0,1,2,3'；默认 auto")
    ap.add_argument("--include-models", default="", help="仅评这些模型，逗号分隔；留空=全部")
    ap.add_argument("--exclude-models", default="", help="排除这些模型，逗号分隔")

    ap.add_argument("--force", action="store_true", help="强制重跑（否则复用已有 JSON）")
    ap.add_argument("--limit", type=int, default=0, help="每个集合最多处理前 N 个样本（调试用）")
    ap.add_argument("--steps", default="both", choices=["extract","score","both"], help="仅抽取/仅计分/全流程")
    ap.add_argument("--scores-out-dir", type=Path, default=None, help="计分输出目录（不传则用 batch_scoring.py 默认）")

    ap.add_argument("--verify", action="store_true", help="启用抽取后绑定核查")
    ap.add_argument("--live", action="store_true", help="终端实时进度（去交错显示，需 tqdm）")
    ap.add_argument("--serialize-output", action="store_true", help="串行执行各 GPU 分桶，完全无交错（牺牲并行度）")

    ap.add_argument("--extra-evi-opts", nargs=argparse.REMAINDER, help="透传给 evi_extract.py 的其他参数")

    # ★ 新增：静默模式，只保留必要信息 & 进度条
    ap.add_argument("--quiet", action="store_true", help="减少日志，仅保留必要信息（建议一键评测脚本使用）")

    return ap.parse_args()

def main():
    args = parse_args()
    gpus = detect_gpus(args.gpus)
    include_models = [x for x in (args.include_models or "").split(",") if x.strip()]
    exclude_models = [x for x in (args.exclude_models or "").split(",") if x.strip()]
    quiet = bool(args.quiet)

    # 抽取
    if args.steps in ("extract", "both"):
        extract_evidence_dist(
            evi_py=args.evi_extract_py.resolve(),
            model_local_path=args.model_local_path.resolve(),
            ref_video_dir=(args.ref_video_dir.resolve() if args.ref_video_dir is not None else None),
            ref_out_dir=args.ref_out_dir.resolve(),
            gen_video_root=args.gen_video_root.resolve(),
            gen_out_root=args.gen_out_root.resolve(),
            include_models=include_models,
            exclude_models=exclude_models,
            gpus=gpus,
            force=args.force,
            limit=args.limit,
            extra_evi_opts=(args.extra_evi_opts or []),
            verify=args.verify,
            live=args.live,
            serialize_output=args.serialize_output,
            quiet=quiet,
        )

    # 计分
    if args.steps in ("score", "both"):
        score_softalign_per_model(
            batch_scoring_py=args.batch_scoring_py.resolve(),
            yaml_path=args.softalign_yaml.resolve(),
            ref_evi_dir=args.ref_out_dir.resolve(),
            gen_evi_root=args.gen_out_root.resolve(),
            include_models=include_models,
            exclude_models=exclude_models,
            out_dir=(args.scores_out_dir.resolve() if args.scores_out_dir else None),
            limit=args.limit,
            pass_through=None,
            quiet=quiet,
        )

if __name__ == "__main__":
    main()
