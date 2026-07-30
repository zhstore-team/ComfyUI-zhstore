# agnes_nodes.py
import json
import os
import time
import tempfile
import subprocess
import shutil
from typing import Dict, Tuple, Optional

# API 配置
BASE_URL = "https://api.agnes-ai.cn"

# 模块级对话历史存储（按 node_id 索引，同一 ComfyUI 会话内持续有效）
_chat_history_store: Dict[str, list] = {}
import torch
import numpy as np
from PIL import Image

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Agnes] OpenCV未安装，请运行: pip install opencv-python")

# ComfyUI 路径工具
import folder_paths

# 配置文件
PLUGIN_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(PLUGIN_DIR, "agnes_config.json")

def load_agnes_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Agnes] 加载配置文件失败: {e}")
    return {}

def get_api_key() -> Optional[str]:
    config = load_agnes_config()
    return config.get("api_key", "")


def _save_chat_history_to_workflow(history, unique_id, extra_pnginfo, key_suffix=""):
    """将对话历史序列化保存到 workflow 节点数据中，实现工作流持久化"""
    if unique_id is None or extra_pnginfo is None:
        return
    try:
        workflow = extra_pnginfo[0]["workflow"]
        node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])), None)
        if node:
            storage_key = f"_agnes_chat_history{key_suffix}"
            # 保存到 properties（工作流保存时会被序列化到 JSON）
            props = node.get("properties", {})
            props[storage_key] = json.dumps(history, ensure_ascii=False)
            node["properties"] = props
    except Exception:
        pass

def _load_chat_history_from_workflow(unique_id, extra_pnginfo, key_suffix=""):
    """从 workflow 节点数据中恢复对话历史"""
    if unique_id is None or extra_pnginfo is None:
        return None
    try:
        workflow = extra_pnginfo[0]["workflow"]
        node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])), None)
        if node:
            storage_key = f"_agnes_chat_history{key_suffix}"
            # 优先从 properties 读取（保存时写入的位置）
            props = node.get("properties", {})
            data = props.get(storage_key) or node.get(storage_key)
            if data:
                return json.loads(data)
    except Exception:
        pass
    return None

def _strip_images_from_history(history):
    """去除对话历史中的图片数据 URI，仅保留文本内容用于持久化存储"""
    stripped = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
            new_content = " ".join(text_parts) if text_parts else "[图像]"
            stripped.append({"role": msg["role"], "content": new_content})
        else:
            stripped.append(msg)
    return stripped


def _get_api_error(status_code: Optional[int], response_body: str = "") -> str:
    """根据 HTTP 状态码返回中文错误说明"""
    error_map = {
        400: "请求参数错误，请检查输入参数是否正确",
        401: "API Key 无效或未授权，请在设置中检查您的 Agnes API Key",
        402: "账户余额不足，请充值后重试",
        403: "无权访问该资源，请检查 API Key 权限",
        404: "请求的资源或接口不存在，请检查请求地址",
        405: "请求方法不允许",
        408: "请求超时，请检查网络连接",
        409: "请求冲突，请稍后重试",
        413: "请求数据过大，请减小输入尺寸或内容",
        415: "不支持的媒体类型",
        422: "请求参数校验失败，请检查输入参数格式",
        429: "请求过于频繁，请稍后重试",
        498: "API Key 无效或已过期",
        499: "请求已关闭（客户端断开连接）",
        500: "服务器内部错误，请稍后重试",
        502: "网关错误，请稍后重试",
        503: "服务暂不可用，请稍后重试",
        504: "网关超时，请稍后重试",
    }
    msg = error_map.get(status_code, f"HTTP 状态码 {status_code}") if status_code else "网络请求失败，请检查网络连接"
    if response_body:
        # 截断过长的响应体
        body = response_body[:600]
        return f"{msg}\n\n响应详情：{body}"
    return msg

# 有效帧数
VALID_NUM_FRAMES = [81, 121, 161, 201, 241, 281, 321, 361, 401, 441]
MIN_FRAMES = min(VALID_NUM_FRAMES)
MAX_FRAMES = max(VALID_NUM_FRAMES)

def duration_to_frames(duration_seconds: float, frame_rate: float) -> int:
    raw_frames = int(round(duration_seconds * frame_rate))
    raw_frames = max(MIN_FRAMES, min(MAX_FRAMES, raw_frames))
    best_frames = min(VALID_NUM_FRAMES, key=lambda x: abs(x - raw_frames))
    return best_frames


class AgnesTextToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a cat walking on the beach at sunset, soft ocean waves, warm golden lighting, realistic motion",
                    "tooltip": "视频内容的文本描述"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Text, watermark, texture image, low quality, blurry, distorted, ugly, bad anatomy, worst quality",
                    "tooltip": "负向提示词"
                }),
                "width": ("INT", {
                    "default": 1152,
                    "min": 256,
                    "max": 1920,
                    "step": 64,
                    "tooltip": "视频宽度"
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1920,
                    "step": 64,
                    "tooltip": "视频高度"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "视频帧率（FPS）"
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 3,
                    "max": 18.0,
                    "step": 0.5,
                    "tooltip": "期望的视频时长（秒），实际时长会略有调整"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，-1表示随机"
                }),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "推理步数"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "generate_video"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Video-V2.0根据文本生成视频。通过时长（秒）控制视频长度，自动适配API帧数约束。需要先在ComfyUI设置中配置 Agens的 API Key。"

    def generate_video(self, prompt: str, negative_prompt: str, width: int, height: int,
                       frame_rate: float, duration_seconds: float, seed: int,
                       num_inference_steps: int = 50) -> Tuple[torch.Tensor, Optional[Dict], float]:
        api_key = get_api_key()
        print("发起请求...")
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        num_frames = duration_to_frames(duration_seconds, frame_rate)
        actual_duration = num_frames / frame_rate
        print(f"[Agnes] 目标时长: {duration_seconds}s, 帧率: {frame_rate}fps -> 使用帧数: {num_frames}, 实际时长: {actual_duration:.2f}s")

        import requests

        create_url = f"{BASE_URL}/v1/videos"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": num_inference_steps,
            "seed":seed
        }
        # if seed != -1:
            # payload["seed"] = seed
        print(seed)

        # 创建任务（只尝试一次，不重试）
        try:
            response = requests.post(create_url, json=payload, headers=headers, timeout=(5, 180))
            print(response)
            response.raise_for_status()
            task_data = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"创建任务超时（180秒），请检查网络或稍后重试")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if (hasattr(e, 'response') and e.response is not None) else None
            response_body = e.response.text[:600] if (hasattr(e, 'response') and e.response is not None) else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"创建任务失败：{error_msg}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有 video_id：{str(task_data)[:300]}")

        print(f"[Agnes] 任务已创建，video_id: {video_id}")

        # 轮询结果（间隔10秒，符合≤20次/分钟的要求）
        video_url = self._poll_for_result(video_id, api_key)
        video_path = self._download_video(video_url)

        frames_tensor, audio_data = self._extract_frames_and_audio(video_path, num_frames)

        try:
            os.unlink(video_path)
        except:
            pass

        return frames_tensor, audio_data, float(frame_rate)

    def _poll_for_result(self, video_id: str, api_key: str, max_retries: int = 180, interval: float = 5.0) -> str:
        """
        轮询查询视频生成结果。
        间隔5秒，总超时30分钟（180次），每分钟12次请求，符合≤20次/分钟的限制。
        """
        import requests
        query_url = f"{BASE_URL}/agnesapi?video_id={video_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        from comfy.utils import ProgressBar
        pbar = ProgressBar(100)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)

                # 用 API 返回的实际进度更新进度条（0~100）
                pbar.update_absolute(progress)

                # 进度卡死检测
                if progress == last_progress and status == "in_progress":
                    stuck_count += 1
                    if stuck_count >= 10:
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed" or result.get("internal_status") == "completed":
                    print(f"[Agnes] 视频生成完成，进度: {progress}%")
                    video_url = result.get("url")
                    if not video_url:
                        metadata = result.get("metadata", {}) or {}
                        video_url = metadata.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL：{str(result)[:300]}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败，原因：{error_msg}")
                else:
                    # queued / in_progress / processing / 未知状态
                    print(f"[Agnes] 状态: {status}, 进度: {progress}%")
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错（HTTP {e.response.status_code if (hasattr(e, 'response') and e.response is not None) else 'N/A'}）: {e}, 重试中...")
                time.sleep(interval)
        raise RuntimeError(f"视频生成超时（已等待 {max_retries * interval} 秒），video_id: {video_id}")

    def _download_video(self, url: str) -> str:
        import requests
        # 使用 ComfyUI 的临时目录
        temp_dir = folder_paths.get_temp_directory()
        temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        print(f"[Agnes] 下载视频: {url}")
        # 增加下载重试（最多2次）
        max_download_retries = 2
        for attempt in range(max_download_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=(10, 120))
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[Agnes] 下载完成: {temp_path}")
                return temp_path
            except Exception as e:
                if attempt < max_download_retries:
                    print(f"[Agnes] 下载失败 ({attempt+1}/{max_download_retries+1})，5秒后重试: {e}")
                    time.sleep(5)
                else:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise RuntimeError(f"下载视频失败：{str(e)}")

    def _extract_frames_and_audio(self, video_path: str, expected_frames: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV未安装，请运行: pip install opencv-python")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            frames.append(img_tensor)
        cap.release()

        if not frames:
            raise RuntimeError("未提取到任何帧")
        frames_tensor = torch.stack(frames, dim=0)
        print(f"[Agnes] 提取帧数: {len(frames)}，张量形状: {frames_tensor.shape}")

        audio_data = self._extract_audio(video_path)
        return frames_tensor, audio_data

    def _extract_audio(self, video_path: str) -> Optional[Dict]:
        """
        使用 imageio_ffmpeg + ffmpeg 命令行提取音频为 WAV，
        然后用 wave 模块直接读取，输出波形形状为 [1, 1, samples]
        以便兼容某些期望三维输入的下游节点（如 combine_video）
        """
        try:
            import imageio_ffmpeg
            import subprocess
            import wave
            import numpy as np

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[Agnes] 使用 FFmpeg 路径: {ffmpeg_exe}")

            # 创建临时 WAV 文件
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()

            # 调用 ffmpeg 提取音频（单声道，16kHz，PCM 16-bit）
            cmd = [
                ffmpeg_exe, "-i", video_path,
                "-vn",                     # 无视频
                "-acodec", "pcm_s16le",   # PCM 16-bit
                "-ar", "16000",           # 16kHz
                "-ac", "1",               # 单声道
                "-y",                     # 覆盖输出
                temp_audio.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[Agnes] ffmpeg 提取失败，返回码: {result.returncode}")
                print(f"[Agnes] stderr: {result.stderr}")
                return None

            # 使用 wave 模块读取 WAV 文件
            with wave.open(temp_audio.name, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)
                # 转换为 float32 范围 [-1, 1]
                waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                # 转换为 torch tensor，形状调整为 [1, 1, samples]（batch, channels, time）
                waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]

            # 截取前30秒
            max_samples = sample_rate * 30
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :, :max_samples]

            print(f"[Agnes] 音频提取成功 (通过 FFmpeg + wave)，波形形状: {waveform.shape}")
            return {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            print(f"[Agnes] 音频提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            # 清理临时文件
            if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)




class AgnesImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入的单张图片，将用于生成动画视频"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Animate the image with subtle motion, natural movement, cinematic quality",
                    "tooltip": "描述图片中需要动画化的内容，例如角色动作、镜头运动等"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Text, watermark, texture image, low quality, blurry, distorted, ugly, bad anatomy, worst quality",
                    "tooltip": "负向提示词，描述需要避免的内容"
                }),
                "width": ("INT", {
                    "default": 1152,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "视频宽度"
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "视频高度"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "视频帧率（FPS）"
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 3,
                    "max": 18.0,
                    "step": 0.5,
                    "tooltip": "期望的视频时长（秒），实际时长会略有调整"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子"
                }),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "推理步数"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "generate_video"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Video-V2.0根据单张图片生成视频。通过时长（秒）控制视频长度，自动适配API帧数约束。需要先在ComfyUI设置中配置Agens的API Key。"

    def generate_video(self, image: torch.Tensor, prompt: str, negative_prompt: str,
                       width: int, height: int, frame_rate: float, duration_seconds: float,
                       seed: int, num_inference_steps: int = 50) -> Tuple[torch.Tensor, Optional[Dict], float]:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # 检查图片尺寸，超限则报错（用最后两个维度取宽高，兼容不同张量形状）
        img_h, img_w = image.shape[-2], image.shape[-1]
        MAX_SIDE = 2048
        if max(img_w, img_h) > MAX_SIDE:
            raise ValueError(f"上传图像尺寸过大（{img_w}x{img_h}），请调整图像尺寸。建议最大边长不超过{MAX_SIDE}像素。")

        # 将输入图像转换为 Data URI Base64 字符串（API 要求 URL 格式）
        image_base64 = self._image_to_data_url(image)
        print(f"[Agnes] 图像已转换为 Data URI (长度: {len(image_base64)})")

        num_frames = duration_to_frames(duration_seconds, frame_rate)
        actual_duration = num_frames / frame_rate
        print(f"[Agnes] 目标时长: {duration_seconds}s, 帧率: {frame_rate}fps -> 使用帧数: {num_frames}, 实际时长: {actual_duration:.2f}s")

        import requests

        create_url = f"{BASE_URL}/v1/videos"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image_base64,          # 使用纯 Base64
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": num_inference_steps,
            "seed":seed
        }
        # if seed != -1:
            # payload["seed"] = seed

        # 创建任务（只尝试一次，不重试）
        try:
            response = requests.post(create_url, json=payload, headers=headers, timeout=(5, 180))
            response.raise_for_status()
            task_data = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("创建任务超时（180秒），请检查网络或稍后重试")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if (hasattr(e, 'response') and e.response is not None) else None
            response_body = e.response.text[:600] if (hasattr(e, 'response') and e.response is not None) else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"创建任务失败：{error_msg}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有 video_id：{str(task_data)[:300]}")

        print(f"[Agnes] 任务已创建，video_id: {video_id}")

        # 轮询结果
        video_url = self._poll_for_result(video_id, api_key)
        video_path = self._download_video(video_url)

        frames_tensor, audio_data = self._extract_frames_and_audio(video_path, num_frames)

        try:
            os.unlink(video_path)
        except:
            pass

        return frames_tensor, audio_data, float(frame_rate)

    def _image_to_data_url(self, image_tensor: torch.Tensor) -> str:
        """
        将 ComfyUI 的图像 Tensor (B,H,W,C) 转换为 Base64 Data URL。
        若 batch 大于 1，默认取第一张图。
        """
        import base64
        from io import BytesIO

        # 取第一张图，并转换为 PIL Image
        if image_tensor.dim() == 4:
            img_tensor = image_tensor[0]          # [H, W, C]
        else:
            img_tensor = image_tensor
        # 确保值域 0-1 并转为 0-255
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')

        # 保存为 PNG 并转 Base64
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"
        
    def _image_to_base64(self, image_tensor: torch.Tensor) -> str:
        """
        将 ComfyUI 的图像 Tensor (B,H,W,C) 转换为纯 Base64 字符串（无 data URL 前缀）。
        若 batch 大于 1，默认取第一张图。
        """
        import base64
        from io import BytesIO

        # 取第一张图
        if image_tensor.dim() == 4:
            img_tensor = image_tensor[0]          # [H, W, C]
        else:
            img_tensor = image_tensor

        # 确保值域 0-1 并转为 0-255
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')

        # 保存为 PNG 并转 Base64（无前缀）
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return b64_str

    def _poll_for_result(self, video_id: str, api_key: str, max_retries: int = 180, interval: float = 5.0) -> str:
        """轮询查询视频结果（与文生视频完全一致）"""
        import requests
        query_url = f"{BASE_URL}/agnesapi?video_id={video_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        from comfy.utils import ProgressBar
        pbar = ProgressBar(100)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)

                # 用 API 返回的实际进度更新进度条（0~100）
                pbar.update_absolute(progress)

                if progress == last_progress and status == "in_progress":
                    stuck_count += 1
                    if stuck_count >= 10:
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed" or result.get("internal_status") == "completed":
                    print(f"[Agnes] 视频生成完成，进度: {progress}%")
                    video_url = result.get("url")
                    if not video_url:
                        metadata = result.get("metadata", {}) or {}
                        video_url = metadata.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL：{str(result)[:300]}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败，原因：{error_msg}")
                else:
                    print(f"[Agnes] 状态: {status}, 进度: {progress}%")
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错（HTTP {e.response.status_code if (hasattr(e, 'response') and e.response is not None) else 'N/A'}）: {e}, 重试中...")
                time.sleep(interval)
        raise RuntimeError(f"视频生成超时（已等待 {max_retries * interval} 秒），video_id: {video_id}")

    def _download_video(self, url: str) -> str:
        """下载视频（与文生视频完全一致）"""
        import requests
        temp_dir = folder_paths.get_temp_directory()
        temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        print(f"[Agnes] 下载视频: {url}")
        max_download_retries = 2
        for attempt in range(max_download_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=(10, 120))
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[Agnes] 下载完成: {temp_path}")
                return temp_path
            except Exception as e:
                if attempt < max_download_retries:
                    print(f"[Agnes] 下载失败 ({attempt+1}/{max_download_retries+1})，5秒后重试: {e}")
                    time.sleep(5)
                else:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise RuntimeError(f"下载视频失败：{str(e)}")

    def _extract_frames_and_audio(self, video_path: str, expected_frames: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        """提取视频帧和音频（与文生视频完全一致）"""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV未安装，请运行: pip install opencv-python")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            frames.append(img_tensor)
        cap.release()

        if not frames:
            raise RuntimeError("未提取到任何帧")
        frames_tensor = torch.stack(frames, dim=0)
        print(f"[Agnes] 提取帧数: {len(frames)}，张量形状: {frames_tensor.shape}")

        audio_data = self._extract_audio(video_path)
        return frames_tensor, audio_data

    def _extract_audio(self, video_path: str) -> Optional[Dict]:
        """从视频中提取音频（与文生视频完全一致）"""
        try:
            import imageio_ffmpeg
            import subprocess
            import wave
            import numpy as np

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[Agnes] 使用 FFmpeg 路径: {ffmpeg_exe}")

            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()

            cmd = [
                ffmpeg_exe, "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", temp_audio.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[Agnes] ffmpeg 提取失败，返回码: {result.returncode}")
                return None

            with wave.open(temp_audio.name, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)
                waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)

            max_samples = sample_rate * 30
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :, :max_samples]

            print(f"[Agnes] 音频提取成功，波形形状: {waveform.shape}")
            return {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            print(f"[Agnes] 音频提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)


class AgnesMultiImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image1": ("IMAGE", {"tooltip": "首帧参考图（必填）"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Smoothly transition from the first keyframe to the last keyframe with natural motion. Start with the first keyframe, gradually transform through natural movement, and end at the last keyframe. Both keyframes should be equally prominent in the video.",
                    "tooltip": "描述首帧到尾帧的过渡动作，如：从A状态平滑变化到B状态"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "low quality, blurry, distorted, ugly, bad anatomy, worst quality",
                    "tooltip": "负向提示词"
                }),
                "width": ("INT", {
                    "default": 1152,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "视频宽度"
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "视频高度"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "视频帧率（FPS）"
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 3,
                    "max": 18.0,
                    "step": 0.5,
                    "tooltip": "期望的视频时长（秒）"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子"
                }),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "推理步数"
                }),
                "image2": ("IMAGE", {"tooltip": "中帧参考图（可选）"}),
                "image3": ("IMAGE", {"tooltip": "尾帧参考图（可选，需至少连1张）"}),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "generate_video"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Video-V2.0根据首帧、尾帧（可选中帧）参考图生成视频，在参考图之间产生平滑过渡动画。"

    def generate_video(self, image1: torch.Tensor, prompt: str, negative_prompt: str,
                       width: int, height: int, frame_rate: float, duration_seconds: float,
                       seed: int, num_inference_steps: int = 50,
                       image2=None, image3=None) -> Tuple[torch.Tensor, Optional[Dict], float]:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # 收集非空图像
        image_list = [image1]
        for img in [image2, image3]:
            if img is not None:
                image_list.append(img)

        # 限制最多3张（API限制）
        if len(image_list) > 3:
            print(f"[Agnes] 警告：API最多支持3张参考图，已截取前3张（共收到{len(image_list)}张）")
            image_list = image_list[:3]

        if len(image_list) < 2:
            raise ValueError(f"至少需要首帧和尾帧2张参考图，当前只有{len(image_list)}张。如只需1张图，请使用「Agnes 单图生视频」节点。")

        # 检查图片尺寸，超限则报错（用最后两个维度取宽高，兼容不同张量形状）
        MAX_SIDE = 2048
        for i, img in enumerate(image_list):
            h, w = img.shape[-2], img.shape[-1]
            if max(w, h) > MAX_SIDE:
                raise ValueError(f"上传图像尺寸过大（{w}x{h}），请调整图像尺寸。建议最大边长不超过{MAX_SIDE}像素。")

        # 将图像列表转换为 Data URI 列表（API 要求 URL 格式）
        image_base64_list = [f"data:image/png;base64,{self._image_to_base64(img)}" for img in image_list]
        print(f"[Agnes] 已转换 {len(image_base64_list)} 张参考图为 Data URI")

        num_frames = duration_to_frames(duration_seconds, frame_rate)
        actual_duration = num_frames / frame_rate
        print(f"[Agnes] 目标时长: {duration_seconds}s, 帧率: {frame_rate}fps -> 使用帧数: {num_frames}, 实际时长: {actual_duration:.2f}s")

        import requests

        create_url = f"{BASE_URL}/v1/videos"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        # 按官方文档：关键帧模式使用 extra_body.image（URL 数组）+ mode=keyframes
        extra_body = {
            "image": image_base64_list,
            "mode": "keyframes",
        }

        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "extra_body": extra_body,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }
        # if seed != -1:
            # payload["seed"] = seed

        # 创建任务
        try:
            response = requests.post(create_url, json=payload, headers=headers, timeout=(5, 180))
            response.raise_for_status()
            task_data = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("创建任务超时（180秒），请检查网络或稍后重试")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if (hasattr(e, 'response') and e.response is not None) else None
            response_body = e.response.text[:600] if (hasattr(e, 'response') and e.response is not None) else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"创建任务失败：{error_msg}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有 video_id：{str(task_data)[:300]}")

        print(f"[Agnes] 任务已创建，video_id: {video_id}")

        # 轮询结果
        video_url = self._poll_for_result(video_id, api_key)
        video_path = self._download_video(video_url)

        frames_tensor, audio_data = self._extract_frames_and_audio(video_path, num_frames)

        try:
            os.unlink(video_path)
        except:
            pass

        return frames_tensor, audio_data, float(frame_rate)

    def _image_to_base64(self, image_tensor: torch.Tensor) -> str:
        """将 ComfyUI 图像 Tensor 转换为纯 Base64 字符串"""
        import base64
        from io import BytesIO

        if image_tensor.dim() == 4:
            img_tensor = image_tensor[0]
        else:
            img_tensor = image_tensor

        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return b64_str

    # 以下方法复制自文生视频节点（完全一致）
    def _poll_for_result(self, video_id: str, api_key: str, max_retries: int = 180, interval: float = 5.0) -> str:
        """轮询查询视频结果"""
        import requests
        query_url = f"{BASE_URL}/agnesapi?video_id={video_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        from comfy.utils import ProgressBar
        pbar = ProgressBar(100)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)

                # 用 API 返回的实际进度更新进度条（0~100）
                pbar.update_absolute(progress)

                if progress == last_progress and status == "in_progress":
                    stuck_count += 1
                    if stuck_count >= 10:
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed" or result.get("internal_status") == "completed":
                    print(f"[Agnes] 视频生成完成，进度: {progress}%")
                    video_url = result.get("url")
                    if not video_url:
                        metadata = result.get("metadata", {}) or {}
                        video_url = metadata.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL：{str(result)[:300]}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败，原因：{error_msg}")
                else:
                    print(f"[Agnes] 状态: {status}, 进度: {progress}%")
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错（HTTP {e.response.status_code if (hasattr(e, 'response') and e.response is not None) else 'N/A'}）: {e}, 重试中...")
                time.sleep(interval)
        raise RuntimeError(f"视频生成超时（已等待 {max_retries * interval} 秒），video_id: {video_id}")

    def _download_video(self, url: str) -> str:
        """下载视频"""
        import requests
        temp_dir = folder_paths.get_temp_directory()
        temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        print(f"[Agnes] 下载视频: {url}")
        max_download_retries = 2
        for attempt in range(max_download_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=(10, 120))
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[Agnes] 下载完成: {temp_path}")
                return temp_path
            except Exception as e:
                if attempt < max_download_retries:
                    print(f"[Agnes] 下载失败 ({attempt+1}/{max_download_retries+1})，5秒后重试: {e}")
                    time.sleep(5)
                else:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise RuntimeError(f"下载视频失败：{str(e)}")

    def _extract_frames_and_audio(self, video_path: str, expected_frames: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        """提取视频帧和音频"""
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV未安装，请运行: pip install opencv-python")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            frames.append(img_tensor)
        cap.release()

        if not frames:
            raise RuntimeError("未提取到任何帧")
        frames_tensor = torch.stack(frames, dim=0)
        print(f"[Agnes] 提取帧数: {len(frames)}，张量形状: {frames_tensor.shape}")

        audio_data = self._extract_audio(video_path)
        return frames_tensor, audio_data

    def _extract_audio(self, video_path: str) -> Optional[Dict]:
        """提取音频"""
        try:
            import imageio_ffmpeg
            import subprocess
            import wave
            import numpy as np

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()

            cmd = [
                ffmpeg_exe, "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", temp_audio.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[Agnes] ffmpeg 提取失败，返回码: {result.returncode}")
                return None

            with wave.open(temp_audio.name, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)
                waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)

            max_samples = sample_rate * 30
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :, :max_samples]

            print(f"[Agnes] 音频提取成功，波形形状: {waveform.shape}")
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[Agnes] 音频提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)



class AgnesTextToImage:
    """文生图节点：根据文本提示生成图像"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a cat walking on the beach at sunset, soft ocean waves, warm golden lighting, realistic motion",
                    "tooltip": "图像内容的文本描述"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "图像宽度"
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "图像高度"
                }),
                   "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Image-2.1-Flash根据文本生成图像。需要先在ComfyUI设置中配置Agens的API Key。"

    def generate_image(self, prompt: str, width: int, height: int, seed: int):
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # 尺寸格式
        size = f"{width}x{height}"
        print(f"[Agnes] 文生图请求: prompt={prompt[:60]}..., size={size}")

        import requests
        url = f"{BASE_URL}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体（文生图，URL输出，然后下载，或者直接Base64）
        # 为避免下载额外URL，使用 return_base64: true 直接获取Base64数据
        payload = {
            "model": "agnes-image-2.1-flash",
            "prompt": prompt,
            "size": size,
            "return_base64": True,   # 直接返回Base64，省去下载步骤
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 300))
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("图像生成请求超时（300秒），请检查网络")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if (hasattr(e, 'response') and e.response is not None) else None
            response_body = e.response.text[:600] if (hasattr(e, 'response') and e.response is not None) else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"图像生成请求失败：{error_msg}")

        # 解析Base64图片数据
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"响应中没有 data 字段：{str(result)[:300]}")
        b64_json = data[0].get("b64_json")
        if not b64_json:
            # 降级：尝试url输出
            img_url = data[0].get("url")
            if img_url:
                print(f"[Agnes] 响应为URL，将下载图片: {img_url}")
                img_data = self._download_image(img_url)
            else:
                raise RuntimeError(f"响应中没有图片数据：{str(result)[:300]}")
        else:
            import base64
            img_data = base64.b64decode(b64_json)

        # 转换为PIL Image并调整尺寸（确保与请求尺寸一致，但API通常返回一致）
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        # 如果尺寸不符，进行缩放（保持比例裁剪？这里简单缩放至目标尺寸）
        if pil_img.size != (width, height):
            pil_img = pil_img.resize((width, height), Image.LANCZOS)

        # 转换为ComfyUI IMAGE张量 [1, H, W, C]
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]

        print(f"[Agnes] 图像生成完成，尺寸: {img_tensor.shape}")
        return (img_tensor,)

    def _download_image(self, url: str) -> bytes:
        """下载图片返回二进制数据"""
        import requests
        response = requests.get(url, timeout=(10, 60))
        response.raise_for_status()
        return response.content


class AgnesImageToImage:
    """图生图节点：基于输入图像和提示词生成新图像"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入图像，将作为参考进行编辑或转换"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform the scene into a rain-soaked cyberpunk night with neon reflections while preserving the original composition",
                    "tooltip": "描述需要如何修改图像，例如风格转换、添加元素等"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "输出图像宽度"
                }),
                "height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "输出图像高度"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Image-2.1-Flash根据输入图像和提示词生成新图像（图生图）。需要配置API Key。"

    def generate_image(self, image: torch.Tensor, prompt: str, width: int, height: int, seed: int):
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # 将输入图像转为Data URI Base64
        image_base64 = self._image_to_data_uri(image)
        print(f"[Agnes] 输入图像已转换为Data URI Base64 (长度: {len(image_base64)})")

        size = f"{width}x{height}"
        print(f"[Agnes] 图生图请求: prompt={prompt[:60]}..., size={size}")

        import requests
        url = f"{BASE_URL}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 构建图生图payload（使用extra_body包含image和response_format）
        extra_body = {
            "image": [image_base64],
            "response_format": "b64_json"   # 要求Base64输出，便于直接解码
        }
        payload = {
            "model": "agnes-image-2.1-flash",
            "prompt": prompt,
            "size": size,
            "extra_body": extra_body
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 300))
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("图生图请求超时（300秒），请检查网络")
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if (hasattr(e, 'response') and e.response is not None) else None
            response_body = e.response.text[:600] if (hasattr(e, 'response') and e.response is not None) else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"图生图请求失败：{error_msg}")

        # 解析Base64图片数据
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"响应中没有 data 字段：{str(result)[:300]}")
        b64_json = data[0].get("b64_json")
        if not b64_json:
            # 降级：尝试url
            img_url = data[0].get("url")
            if img_url:
                print(f"[Agnes] 响应为URL，将下载图片: {img_url}")
                img_data = self._download_image(img_url)
            else:
                raise RuntimeError(f"响应中没有图片数据：{str(result)[:300]}")
        else:
            import base64
            img_data = base64.b64decode(b64_json)

        # 转换为PIL并调整尺寸
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if pil_img.size != (width, height):
            pil_img = pil_img.resize((width, height), Image.LANCZOS)

        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        print(f"[Agnes] 图生图完成，输出尺寸: {img_tensor.shape}")
        return (img_tensor,)

    def _image_to_data_uri(self, image_tensor: torch.Tensor) -> str:
        """将ComfyUI图像张量转换为Data URI Base64字符串（格式：data:image/png;base64,xxx）"""
        import base64
        from io import BytesIO
        from PIL import Image

        # 取第一张图，形状 [H, W, C]
        if image_tensor.dim() == 4:
            img_tensor = image_tensor[0]
        else:
            img_tensor = image_tensor
        # 值域 0-1 转 0-255
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"

    def _download_image(self, url: str) -> bytes:
        """下载图片二进制数据"""
        import requests
        response = requests.get(url, timeout=(10, 60))
        response.raise_for_status()
        return response.content


# class AgnesMultiImageToImage:
    # """多图编辑节点：基于多张参考图像和提示词生成新图像（最多10张，第1张必填）"""
    # @classmethod
    # def INPUT_TYPES(cls):
        # return {
            # "required": {
                # "prompt": ("STRING", {
                    # "multiline": True,
                    # "default": "Combine the characteristics of the provided images into a single coherent scene, maintaining visual consistency and high quality",
                    # "tooltip": "正面提示词：描述期望生成的图像内容，尤其可说明多张图片如何融合"
                # }),
                # "image1": ("IMAGE", {
                    # "tooltip": "第一张参考图像（必填）"
                # }),
                # "width": ("INT", {
                    # "default": 1024,
                    # "min": 256,
                    # "max": 1920,
                    # "step": 64,
                    # "tooltip": "输出图像宽度"
                # }),
                # "height": ("INT", {
                    # "default": 768,
                    # "min": 256,
                    # "max": 1920,
                    # "step": 64,
                    # "tooltip": "输出图像高度"
                # }),
                # "seed": ("INT", {
                    # "default": 0,               # 改为 -1 表示自动随机
                    # "min": 0,
                    # "max": 0xffffffffffffffff,
                    # "tooltip": "随机种子"
                # }),
            # },
            # "optional": {
                # "image2": ("IMAGE", {"tooltip": "第二张参考图像（可选）"}),
                # "image3": ("IMAGE", {"tooltip": "第三张参考图像（可选）"}),
                # "image4": ("IMAGE", {"tooltip": "第四张参考图像（可选）"}),
                # "image5": ("IMAGE", {"tooltip": "第五张参考图像（可选）"}),
                # "image6": ("IMAGE", {"tooltip": "第六张参考图像（可选）"}),
                # "image7": ("IMAGE", {"tooltip": "第七张参考图像（可选）"}),
                # "image8": ("IMAGE", {"tooltip": "第八张参考图像（可选）"}),
                # "image9": ("IMAGE", {"tooltip": "第九张参考图像（可选）"}),
                # "image10": ("IMAGE", {"tooltip": "第十张参考图像（可选）"}),
            # }
        # }

    # RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image",)
    # FUNCTION = "generate_image"
    # CATEGORY = "智绘Store/Agens AI"
    # DESCRIPTION = "使用Agnes-Image-2.1-Flash根据多张参考图像生成新图像（最多10张）。第一张图像必填，其余可选。需要配置API Key。"

    # def generate_image(self, prompt: str, image1: torch.Tensor,
                       # width: int, height: int, seed: int,
                       # image2=None, image3=None, image4=None, image5=None,
                       # image6=None, image7=None, image8=None, image9=None, image10=None):
        # api_key = get_api_key()
        # if not api_key:
            # raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # # 收集所有非空图像
        # images = [image1]
        # for img in [image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            # if img is not None:
                # images.append(img)
        # print(f"[Agnes] 共收集到 {len(images)} 张参考图像")

        # # 将每张图像转换为 Data URI Base64
        # image_data_uris = [self._image_to_data_uri(img) for img in images]
        # print(f"[Agnes] 已转换 {len(image_data_uris)} 张图像为 Data URI")

        # size = f"{width}x{height}"
        # print(f"[Agnes] 多图编辑请求: prompt={prompt[:60]}..., size={size}, 图像数量={len(image_data_uris)}")

        # import requests
        # url = f"{BASE_URL}/v1/images/generations"
        # headers = {
            # "Authorization": f"Bearer {api_key}",
            # "Content-Type": "application/json"
        # }

        # # 构建 extra_body，包含多张图像数组和 response_format
        # extra_body = {
            # "image": image_data_uris,       # 数组形式，包含所有参考图像
            # "response_format": "b64_json",   # 要求 Base64 输出
            
        # }

        # payload = {
            # "model": "agnes-image-2.1-flash",
            # "prompt": prompt,
            # "size": size,
            # "extra_body": extra_body
       
        # }

        # try:
            # response = requests.post(url, json=payload, headers=headers, timeout=(10, 180))
            # response.raise_for_status()
            # result = response.json()
        # except requests.exceptions.Timeout:
            # raise RuntimeError("多图编辑请求超时（180秒），请检查网络")
        # except requests.exceptions.RequestException as e:
            # raise RuntimeError(f"多图编辑请求失败: {str(e)}")

        # # 解析 Base64 图片数据
        # data = result.get("data", [])
        # if not data:
            # raise RuntimeError(f"响应中没有 data 字段: {result}")
        # b64_json = data[0].get("b64_json")
        # if not b64_json:
            # img_url = data[0].get("url")
            # if img_url:
                # print(f"[Agnes] 响应为 URL，将下载图片: {img_url}")
                # img_data = self._download_image(img_url)
            # else:
                # raise RuntimeError(f"响应中没有图片数据: {result}")
        # else:
            # import base64
            # img_data = base64.b64decode(b64_json)

        # # 转换为 PIL 并调整尺寸
        # from PIL import Image
        # import io
        # pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        # if pil_img.size != (width, height):
            # pil_img = pil_img.resize((width, height), Image.LANCZOS)

        # img_np = np.array(pil_img).astype(np.float32) / 255.0
        # img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # print(f"[Agnes] 多图编辑完成，输出尺寸: {img_tensor.shape}")
        # return (img_tensor,)

    # # ---------- 辅助方法 ----------
    # def _image_to_data_uri(self, image_tensor: torch.Tensor) -> str:
        # """将 ComfyUI 图像张量转换为 Data URI Base64 字符串"""
        # import base64
        # from io import BytesIO
        # from PIL import Image

        # if image_tensor.dim() == 4:
            # img_tensor = image_tensor[0]
        # else:
            # img_tensor = image_tensor
        # img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        # pil_img = Image.fromarray(img_np, mode='RGB')
        # buffer = BytesIO()
        # pil_img.save(buffer, format='PNG')
        # b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # return f"data:image/png;base64,{b64_str}"

    # def _download_image(self, url: str) -> bytes:
        # """下载图片二进制数据"""
        # import requests
        # response = requests.get(url, timeout=(10, 60))
        # response.raise_for_status()
        # return response.content

class AgnesChat:
    """
    Agnes 文本对话节点：支持连续对话、流式输出
    - 支持选择模型（agnes-2.5-flash / agnes-2.0-flash）
    - 下方文本框输入用户提示词
    - 可选系统提示词输入端口
    - 对话记忆开关 + 工作流持久化
    - 输出端口输出最新一条回复和格式化对话历史
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["agnes-2.5-flash", "agnes-2.0-flash"], {
                    "default": "agnes-2.5-flash",
                    "tooltip": "选择文本模型，2.5 为最新升级版"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "在此输入您的问题"
                }),
                "thinking_enabled": ("BOOLEAN", {
                    "default": True,
                    "label": "思考模式",
                    "tooltip": "启用后模型会先进行深度推理再给出回答，适用于复杂推理任务"
                }),
                "max_tokens": ("INT", {
                    "default": 64000,
                    "min": 64,
                    "max": 65536,
                    "step": 64,
                    "tooltip": "最大生成 Token 数量"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "控制输出随机性，值越低结果越确定"
                }),
                "memory_enabled": ("BOOLEAN", {
                    "default": True,
                    "label": "对话记忆",
                    "tooltip": "启用后保留对话上下文，支持连续多轮对话。关闭后每次对话独立"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "可选的系统提示词，从输入端口接入"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复", "对话历史")
    FUNCTION = "chat"
    OUTPUT_NODE = True
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "Agnes 文本对话：支持选择模型（默认 agnes-2.5-flash）、连续对话、流式输出。「回复」端口输出最新回复，「对话历史」端口输出格式化后的完整对话。"

    def _format_history_output(self, history: list, system_prompt: str = "") -> str:
        """将对话历史格式化为指定格式的输出文本"""
        lines = []
        if system_prompt and system_prompt.strip():
            lines.append(f"系统提示词：{system_prompt}")
            lines.append("————————————————————————")
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"Agnes：{content}")
            elif role == "system":
                lines.append(f"系统提示词：{content}")
            else:
                lines.append(f"{role}：{content}")
            lines.append("————————————————————————")
        return "\n".join(lines)

    def chat(self, model, user_prompt, thinking_enabled, max_tokens, temperature, memory_enabled=True, system_prompt=None, unique_id=None, extra_pnginfo=None):
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        if not user_prompt or user_prompt.strip() == "":
            raise ValueError("请输入用户提示词")

        node_id = str(unique_id[0]) if unique_id else "default"

        # 获取对话历史（优先从内存取，其次从工作流恢复）
        history = _chat_history_store.get(node_id, [])
        if not history and memory_enabled:
            saved = _load_chat_history_from_workflow(unique_id, extra_pnginfo)
            if saved:
                history = saved
                _chat_history_store[node_id] = history

        # 构建 messages 数组
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        if memory_enabled:
            messages.extend(history)
        if user_prompt and user_prompt.strip():
            messages.append({"role": "user", "content": user_prompt})

        # 调用流式 API
        url = f"{BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # 启用思考模式时添加 chat_template_kwargs（OpenAI 兼容格式）
        if thinking_enabled:
            payload["chat_template_kwargs"] = {
                "thinking": True,
            }

        import requests
        response_text = ""

        try:
            http_response = requests.post(url, json=payload, headers=headers, timeout=(10, 120))
            http_response.raise_for_status()
            result = http_response.json()
            choices = result.get('choices', [])
            if choices:
                response_text = choices[0].get('message', {}).get('content', '').strip()
        except requests.exceptions.Timeout:
            raise RuntimeError("对话请求超时（120秒），请检查网络或稍后重试")
        except requests.exceptions.RequestException as e:
            resp = e.response
            status_code = resp.status_code if resp is not None else None
            response_body = resp.text[:600] if resp is not None else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"对话请求失败：{error_msg}")

        if not response_text:
            raise RuntimeError("API 返回内容为空，请重试")

        # 更新对话历史
        if memory_enabled:
            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": response_text})
            _chat_history_store[node_id] = history
            # 持久化到工作流（不含图片，文本对话无需剥离）
            _save_chat_history_to_workflow(history, unique_id, extra_pnginfo)
        else:
            # 不启用记忆时，仅保留当前轮次用于格式化输出
            history = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
            ]

        # 生成格式化对话历史输出
        formatted_history = self._format_history_output(history, system_prompt)

        # 在 ui 中返回序列化历史，供前端保存到 this.properties 实现持久化
        return {"ui": {"text": response_text, "history": json.dumps(history, ensure_ascii=False)}, "result": (response_text, formatted_history)}


class AgnesVisionChat:
    """
    Agnes 图像理解对话节点：支持图像+文本多模态输入、连续对话
    - 支持选择模型（agnes-2.5-flash / agnes-2.0-flash）
    - 图像输入端口 + 文本提示词
    - 可选系统提示词输入端口
    - 对话记忆开关 + 工作流持久化
    - 输出端口输出最新一条回复和格式化对话历史
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["agnes-2.5-flash", "agnes-2.0-flash"], {
                    "default": "agnes-2.5-flash",
                    "tooltip": "选择文本模型，2.5 支持多模态输入（图像+文本）"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "在此输入您的问题（可结合图像提问）"
                }),
                "thinking_enabled": ("BOOLEAN", {
                    "default": True,
                    "label": "思考模式",
                    "tooltip": "启用后模型会先进行深度推理再给出回答，适用于复杂推理任务"
                }),
                "max_tokens": ("INT", {
                    "default": 64000,
                    "min": 64,
                    "max": 65536,
                    "step": 64,
                    "tooltip": "最大生成 Token 数量"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "控制输出随机性，值越低结果越确定"
                }),
                "memory_enabled": ("BOOLEAN", {
                    "default": False,
                    "label": "对话记忆",
                    "tooltip": "启用后保留对话上下文，支持连续多轮对话。关闭后每次对话独立"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "forceInput": True,
                    "tooltip": "输入图像（可选），支持多张图像组成的批次"
                }),
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "可选的系统提示词，从输入端口接入"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回复", "对话历史")
    FUNCTION = "chat"
    OUTPUT_NODE = True
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "Agnes 图像理解：支持上传图像进行视觉问答、图像描述等。需要 agnes-2.5-flash 模型。「回复」端口输出最新回复，「对话历史」端口输出格式化后的完整对话。"

    def _tensor_to_data_uri(self, image_tensor: torch.Tensor) -> str:
        """将 ComfyUI 图像张量转换为 Data URI Base64 字符串"""
        import base64
        from io import BytesIO
        from PIL import Image

        if image_tensor.dim() == 4:
            img_tensor = image_tensor[0]
        else:
            img_tensor = image_tensor
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"

    def _format_history_output(self, history: list, system_prompt: str = "") -> str:
        """将对话历史格式化为指定格式的输出文本（兼容多模态消息）"""
        lines = []
        if system_prompt and system_prompt.strip():
            lines.append(f"系统提示词：{system_prompt}")
            lines.append("————————————————————————")
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # 多模态内容，提取文本部分
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = (" ".join(text_parts) + " [图像]").strip()
                if content == "[图像]":
                    content = "[用户上传了图像]"
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"Agnes：{content}")
            elif role == "system":
                lines.append(f"系统提示词：{content}")
            else:
                lines.append(f"{role}：{content}")
            lines.append("————————————————————————")
        return "\n".join(lines)

    def chat(self, model, user_prompt, thinking_enabled, max_tokens, temperature, memory_enabled=True, images=None, system_prompt=None, unique_id=None, extra_pnginfo=None):
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        if not user_prompt or user_prompt.strip() == "":
            if images is None:
                raise ValueError("请输入用户提示词")
        node_id = "vision_" + str(unique_id[0]) if unique_id else "vision_default"

        # 获取对话历史（优先从内存取，其次从工作流恢复）
        history = _chat_history_store.get(node_id, [])
        if not history and memory_enabled:
            saved = _load_chat_history_from_workflow(unique_id, extra_pnginfo, key_suffix="_vision")
            if saved:
                history = saved
                _chat_history_store[node_id] = history

        # 构建 messages 数组
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        if memory_enabled:
            messages.extend(history)

        # 构建用户消息 content（支持纯文本或图像+文本多模态）
        content_array = None
        if images is not None:
            content_array = [{"type": "text", "text": user_prompt}]
            if images.dim() == 4:
                batch_size = images.shape[0]
                for i in range(batch_size):
                    img_tensor = images[i].unsqueeze(0) if images.dim() == 4 else images
                    data_uri = self._tensor_to_data_uri(img_tensor)
                    content_array.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
            else:
                data_uri = self._tensor_to_data_uri(images)
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
            messages.append({"role": "user", "content": content_array})
        else:
            messages.append({"role": "user", "content": user_prompt})

        # 调用流式 API
        url = f"{BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # 启用思考模式
        if thinking_enabled:
            payload["chat_template_kwargs"] = {
                "thinking": True,
            }

        import requests
        response_text = ""

        try:
            http_response = requests.post(url, json=payload, headers=headers, timeout=(10, 120))
            http_response.raise_for_status()
            result = http_response.json()
            choices = result.get('choices', [])
            if choices:
                response_text = choices[0].get('message', {}).get('content', '').strip()
        except requests.exceptions.Timeout:
            raise RuntimeError("对话请求超时（120秒），请检查网络或稍后重试")
        except requests.exceptions.RequestException as e:
            resp = e.response
            status_code = resp.status_code if resp is not None else None
            response_body = resp.text[:600] if resp is not None else ""
            error_msg = _get_api_error(status_code, response_body)
            raise RuntimeError(f"对话请求失败：{error_msg}")

        if not response_text:
            raise RuntimeError("API 返回内容为空，请重试")

        # 更新对话历史
        if memory_enabled:
            user_content = content_array if content_array else user_prompt
            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": response_text})
            _chat_history_store[node_id] = history
            # 持久化到工作流（剥离图片数据以减小工作流体积）
            history_for_storage = _strip_images_from_history(history)
            _save_chat_history_to_workflow(history_for_storage, unique_id, extra_pnginfo, key_suffix="_vision")
        else:
            # 不启用记忆时，仅保留当前轮次用于格式化输出
            history = [
                {"role": "user", "content": content_array if content_array else user_prompt},
                {"role": "assistant", "content": response_text},
            ]

        # 生成格式化对话历史输出
        formatted_history = self._format_history_output(history, system_prompt)

        # 在 ui 中返回序列化历史，供前端保存到 this.properties 实现持久化
        return {"ui": {"text": response_text, "history": json.dumps(history, ensure_ascii=False)}, "result": (response_text, formatted_history)}


class AgnesAssistantExpert:
    """
    Agnes 助手专家节点：从 assistant.txt 加载预设的系统提示词，提供下拉选择
    - 输出选中的系统提示词文本，可直接连接到 AgnesChat / AgnesVisionChat 的 system_prompt 端口
    """
    @classmethod
    def INPUT_TYPES(cls):
        prompts = cls._load_prompts()
        options = list(prompts.keys())
        return {
            "required": {
                "assistant": (options, {
                    "default": options[0] if options else "",
                    "tooltip": "选择智能助手角色，获取对应的系统提示词"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("系统提示词",)
    FUNCTION = "get_prompt"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "Agnes 助手专家：提供预设的系统提示词，用于引导文本对话和图像理解对话节点的行为。"

    @staticmethod
    def _load_prompts() -> dict:
        """从 resource/assistant/ 文件夹加载提示词预设，每个 .md 文件为一个选项
        文件名格式：{排序}-{助手名}.md，从文件名解析助手名作为下拉选项标题
        """
        prompts = {}
        folder = os.path.join(PLUGIN_DIR, "resource", "assistant")
        if not os.path.isdir(folder):
            return {"默认助手": "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"}

        # 列出所有 .md 文件并按文件名排序
        files = sorted([f for f in os.listdir(folder) if f.endswith(".md")])
        for filename in files:
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            # 从文件名解析标题：去掉排序前缀和扩展名
            # "3-分镜制作专家.md" -> "分镜制作专家"
            title = filename.rsplit(".", 1)[0]  # 去掉 .md
            if "-" in title:
                title = title.split("-", 1)[1]  # 去掉 "N-" 前缀
            prompts[title] = content

        if not prompts:
            prompts = {"默认助手": "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"}

        return prompts

    def get_prompt(self, assistant):
        prompts = self._load_prompts()
        return (prompts.get(assistant, ""),)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AgnesTextToVideo": AgnesTextToVideo,
    "AgnesImageToVideo": AgnesImageToVideo,
    "AgnesMultiImageToVideo": AgnesMultiImageToVideo, 
    "AgnesTextToImage": AgnesTextToImage,
    "AgnesImageToImage": AgnesImageToImage,
    "AgnesChat": AgnesChat,
    "AgnesVisionChat": AgnesVisionChat,
    "AgnesAssistantExpert": AgnesAssistantExpert,
    # "AgnesMultiImageToImage": AgnesMultiImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgnesTextToVideo": "Agnes 文生视频",
    "AgnesImageToVideo": "Agnes 单图生视频",
    "AgnesMultiImageToVideo": "Agnes 首尾帧生视频",
    "AgnesTextToImage": "Agnes 文生图",
    "AgnesImageToImage": "Agnes 图生图",
    "AgnesChat": "Agnes 文本对话",
    "AgnesVisionChat": "Agnes 图像理解对话",
    "AgnesAssistantExpert": "Agnes 助手专家",
    # "AgnesMultiImageToImage": "Agnes 多图编辑",
}


# ====== 清空对话历史 API 端点 ======
from aiohttp import web
import server

@server.PromptServer.instance.routes.post("/agnes/clear_chat")
async def clear_chat(request):
    """清空指定 node 的对话历史（支持普通 chat 和 vision chat）"""
    try:
        data = await request.json()
        node_id = data.get("node_id", "")
        if not node_id:
            return web.json_response({"status": "error", "message": "node_id is required"}, status=400)
        # 尝试清除普通对话历史和 vision 对话历史
        cleared = False
        for key in [node_id, "vision_" + node_id]:
            if key in _chat_history_store:
                del _chat_history_store[key]
                cleared = True
        return web.json_response({"status": "ok", "cleared": cleared})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)