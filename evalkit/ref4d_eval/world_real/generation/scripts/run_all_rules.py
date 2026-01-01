#!/usr/bin/env python3
import subprocess, shlex, os, datetime, sys

MODEL_PATH = "/root/autodl-tmp/aiv/models/openbmb__MiniCPM-V-4_5"
BASE_EVID  = "/root/autodl-tmp/aiv/evidence"
BASE_VIDEO = "/root/autodl-tmp/aiv/data/refvideo"
BASE_OUT   = "/root/autodl-tmp/aiv/rules"

CATEGORIES = [
    "people_daily",
    "landscape",
]

RULE_SCRIPT = "rule_many.py"  # 按需改为绝对路径

COMMON_ARGS = [
    "--device", "cuda",
    "--dtype", "bf16",
    "--window-sec", "6",
    "--hop-sec", "3",
    "--fps", "3",
    "--cap-frames", "300",
    "--resize-short", "448",
    "--max-new-tokens", "512",
    "--temperature", "0.0",
    "--f-time-unit", "auto",
    "--dump-debug",
    "--verbose",
    "--global-fallback",
    "--global-fallback-threshold", "3",
    "--fallback-whole-video",
    "--fallback-whole-fps", "4",
]

def run_with_retry(cmd_list, log_file, max_tries=3, delay=15):
    tries = 0
    while True:
        tries += 1
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n==== Try {tries} ====\n")
            proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            import time; time.sleep(delay)
        except KeyboardInterrupt:
            return False

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for cat in CATEGORIES:
        json_dir = os.path.join(BASE_EVID, cat)
        video_dir = os.path.join(BASE_VIDEO, cat)
        out_dir  = os.path.join(BASE_OUT, cat)

        if not os.path.isdir(json_dir):
            print(f"[WARN] 缺少证据目录：{json_dir}，跳过该类目。")
            continue
        if not os.path.isdir(video_dir):
            print(f"[WARN] 缺少视频目录：{video_dir}，跳过该类目。")
            continue
        os.makedirs(out_dir, exist_ok=True)

        log_file = os.path.join(out_dir, f"run_{timestamp}.log")
        print(f"======== 开始处理：{cat} ========")
        print(f"日志：{log_file}")

        cmd = [
            sys.executable, RULE_SCRIPT,
            "--json-dir", json_dir,
            "--video-dir", video_dir,
            "--out-dir", out_dir,
            "--local-path", MODEL_PATH,
            *COMMON_ARGS
        ]

        ok = run_with_retry(cmd, log_file)
        if not ok:
            # 不中断全部流程，继续下一个类目
            pass

        print(f"======== 完成：{cat} ========")

    print(f"全部类目处理完成。输出根目录：{BASE_OUT}")

if __name__ == "__main__":
    main()
