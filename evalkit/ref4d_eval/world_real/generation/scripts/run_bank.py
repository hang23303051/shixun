#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行 question_bank.py：
- 读取 data/refvideo/<category> 作为 --video-dir
- 读取 vqa/<category>        作为 --vqa-dir
- 读取 assertion/<category>  作为 --assert-dir
- 输出到 question_bank/<category> 作为 --out-dir
- 失败自动重试，不中断后续类目
"""

import subprocess, os, datetime, sys, time

# ================== 基础路径（按需修改） ==================
MODEL_PATH = "/root/autodl-tmp/aiv/models/openbmb__MiniCPM-V-4_5"
BASE_VIDEO = "/root/autodl-tmp/aiv/data/refvideo"        # --video-dir
BASE_VQA   = "/root/autodl-tmp/aiv/vqa"                  # --vqa-dir
BASE_ASSERT= "/root/autodl-tmp/aiv/assertion"            # --assert-dir
BASE_OUT   = "/root/autodl-tmp/aiv/question_bank"        # --out-dir

# ================== 类目清单（可增删顺序） ==================
CATEGORIES = [
    "landscape",
    "people_daily",
]

# ================== 可执行脚本名（相对路径或绝对路径） ==================
QUESTION_BANK_SCRIPT = "question_bank.py"  # 若不在当前目录，请改成绝对路径

# ================== 公共参数（对应你提供的命令） ==================
COMMON_ARGS = [
    "--local-path", MODEL_PATH,
    "--device", "cuda",
    "--dtype", "bf16",
    "--fps", "3",
    "--cap-frames", "300",
    "--resize-short", "448",
    "--max-new-tokens", "256",
    "--temperature", "0.0",
    "--verbose",
]

# ================== 工具函数 ==================
def run_with_retry(cmd_list, log_file, max_tries=3, delay=15):
    """
    带重试的运行：将 stdout/stderr 同时写到屏幕与日志文件。
    """
    tries = 0
    while True:
        tries += 1
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n==== Try {tries} ====\n")
            f.write("CMD: " + " ".join(cmd_list) + "\n\n")
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            ret = proc.wait()

        if ret == 0:
            return True
        if tries >= max_tries:
            print(f"[FATAL] 命令多次失败，已放弃。详见日志：{log_file}")
            return False

        print(f"[WARN] 运行失败，{delay}s 后重试（{tries}/{max_tries}）...")
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return False

def main():
    # 检查可执行脚本
    if not os.path.isfile(QUESTION_BANK_SCRIPT):
        print(f"[WARN] 找不到脚本：{QUESTION_BANK_SCRIPT}，请确认路径（可改为绝对路径）。")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for cat in CATEGORIES:
        video_dir  = os.path.join(BASE_VIDEO,  cat)
        vqa_dir    = os.path.join(BASE_VQA,    cat)
        assert_dir = os.path.join(BASE_ASSERT, cat)
        out_dir    = os.path.join(BASE_OUT,    cat)

        # 目录检查
        missed = False
        if not os.path.isdir(video_dir):
            print(f"[WARN] 缺少视频目录（--video-dir）：{video_dir}，跳过 {cat}")
            missed = True
        if not os.path.isdir(vqa_dir):
            print(f"[WARN] 缺少 VQA 目录（--vqa-dir）：{vqa_dir}，跳过 {cat}")
            missed = True
        if not os.path.isdir(assert_dir):
            print(f"[WARN] 缺少 assertion 目录（--assert-dir）：{assert_dir}，跳过 {cat}")
            missed = True
        if missed:
            continue

        os.makedirs(out_dir, exist_ok=True)

        # 每类一个日志文件
        log_file = os.path.join(out_dir, f"run_{timestamp}.log")
        print(f"======== 开始处理：{cat} ========")
        print(f"日志：{log_file}")

        # 组装命令
        cmd = [
            sys.executable, QUESTION_BANK_SCRIPT,
            "--video-dir",  video_dir,
            "--vqa-dir",    vqa_dir,
            "--assert-dir", assert_dir,
            "--out-dir",    out_dir,
            *COMMON_ARGS
        ]

        ok = run_with_retry(cmd, log_file, max_tries=3, delay=15)
        if not ok:
            # 不终止后续类目
            pass

        print(f"======== 完成：{cat} ========\n")

    print(f"全部类目处理完成。输出根目录：{BASE_OUT}")

if __name__ == "__main__":
    main()
