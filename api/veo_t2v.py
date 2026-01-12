import argparse
import requests
import json
import time
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------
# 任务轮询
# ----------------------------
def wait_for_video(task_id, query_url, api_key, interval=10, timeout=800):
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Video generation timeout")

        resp = requests.get(
            query_url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            params={"id": task_id}
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        print(f"[Task {task_id}] Status: {status}")

        if status in ("success", "completed"):
            return data["video_url"]
        if status in ("failed", "error"):
            raise RuntimeError(f"Video generation failed: {data}")

        time.sleep(interval)


# ----------------------------
# 下载视频
# ----------------------------
def download_video(video_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    resp = requests.get(video_url, stream=True)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[Saved] {save_path}")


# --------------------------------
# 构造 Prompt（multi / single 逻辑）
# --------------------------------
def build_prompt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()

    filename = os.path.basename(txt_path).lower()

    if "multi" in filename:
        prefix = (
            "This video should contain multiple distinct shots, "
            "with clear scene transitions, varied camera angles, "
            "and coherent visual storytelling.\n\n"
        )
        return prefix + base_prompt

    # single 或其他情况
    return base_prompt


# ----------------------------
# 调用视频生成 API
# ----------------------------
def create_video(prompt, args):
    payload = {
        "prompt": prompt,
        "model": args.model,
        "enhance_prompt": True,
        "enable_upsample": True,
        "aspect_ratio": args.aspect_ratio,
        "resolution": args.resolution,
        "images": []
    }

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        args.api_url,
        headers=headers,
        data=json.dumps(payload)
    )
    resp.raise_for_status()

    data = resp.json()
    return data["id"]


# ----------------------------
# 处理单个 txt 文件
# ----------------------------
def process_txt_file(txt_path, args):
    print(f"\n[Processing] {txt_path}")

    prompt = build_prompt(txt_path)
    task_id = create_video(prompt, args)

    print(f"Task ID: {task_id} → polling...")
    video_url = wait_for_video(
        task_id,
        args.query_url,
        args.api_key,
        timeout=args.timeout
    )

    # 代码从 .txt 的父目录名推断视频类别
    # save_path =  output_dir / category / video_name + ".mp4"
    category = os.path.basename(os.path.dirname(txt_path))
    video_name = os.path.splitext(os.path.basename(txt_path))[0] + ".mp4"
    save_path = os.path.join(args.output_dir, category, video_name)
    download_video(video_url, save_path)


def main():
    parser = argparse.ArgumentParser(description="Text-to-Video generation via LLM API")
    parser.add_argument("--api_url", required=True, help="Video create API URL")
    parser.add_argument("--query_url", required=True, help="Video query API URL")
    parser.add_argument("--api_key", required=True, help="API Key")
    parser.add_argument("--model", default="veo2-fast-components", help="Model name")

    parser.add_argument("--input_path", required=True, help="Prompt .txt file or directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated videos")

    parser.add_argument("--resolution", default="720p", help="Video resolution")
    parser.add_argument("--aspect_ratio", default="16:9", help="Video aspect ratio")
    parser.add_argument("--timeout", type=int, default=800, help="Task timeout (seconds)")

    args = parser.parse_args()

    # 如果是单个 .txt 文件 → 只处理这一个 prompt
    if os.path.isfile(args.input_path):
        process_txt_file(args.input_path, args)

    # 如果是文件夹 → 递归遍历该目录下的所有子目录和文件
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(".txt"):
                    process_txt_file(os.path.join(root, file), args)
    else:
        raise ValueError("input_path must be a .txt file or a directory")
    

if __name__ == "__main__":
    main()

# python veo_t2v.py --api_url https://yunwu.ai/v1/video/create --query_url https://yunwu.ai/v1/video/query --api_key sk-eoPeTYgAG2lC7C5RrOcLNKPYvlAZqCBI9mgWYZuwyd6gJxog --model veo2-fast-components --input_path D:\tv2v --output_dir D:\tv2v\output