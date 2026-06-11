# agnes_nodes.py
import json
import os
import time
import tempfile
import subprocess
import shutil
from typing import Dict, Tuple, Optional
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

        create_url = "https://apihub.agnes-ai.com/v1/videos"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
            raise RuntimeError(f"创建任务失败: {str(e)}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有video_id: {task_data}")

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
        默认间隔3秒，总超时30分钟（180次），每分钟仅6次请求，远低于20次/分钟限制。
        """
        import requests
        query_url = f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}&model_name=agnes-video-v2.0"
        headers = {"Authorization": f"Bearer {api_key}"}

        from comfy.utils import ProgressBar
        pbar = ProgressBar(max_retries)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))  # 连接5秒，读取15秒
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)
                pbar.update_absolute(attempt, max_retries)

                # 美化状态显示
                display_status = status
                if status == "queued":
                    display_status = "排队中"
                elif status == "in_progress":
                    display_status = "视频生成中"
                elif status == "processing":
                    display_status = "处理中"
                elif status == "completed":
                    display_status = "已完成"
                elif status == "failed":
                    display_status = "失败"

                print(f"[Agnes] 状态: {display_status}, 进度: {progress}%")

                # 进度卡死检测（可选，防止无限等待）
                if progress == last_progress and status in ["in_progress", "processing"]:
                    stuck_count += 1
                    if stuck_count >= 10:  # 连续10次，进度不变，视为异常
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        # 不抛出异常，继续等待，但重置计数器避免频繁打印
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed":
                    video_url = result.get("remixed_from_video_id")
                    if not video_url:
                        video_url = result.get("video_url") or result.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL: {result}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败: {error_msg}")
                elif status in ["queued", "in_progress", "processing"]:
                    time.sleep(interval)
                else:
                    # 未知状态，仍等待
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错: {e}, 重试中...")
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
                    raise RuntimeError(f"下载视频失败: {str(e)}")

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

        # 将输入图像转换为纯 Base64 字符串（无前缀）
        image_base64 = self._image_to_base64(image)
        print(f"[Agnes] 图像已转换为 Base64 (长度: {len(image_base64)})")

        num_frames = duration_to_frames(duration_seconds, frame_rate)
        actual_duration = num_frames / frame_rate
        print(f"[Agnes] 目标时长: {duration_seconds}s, 帧率: {frame_rate}fps -> 使用帧数: {num_frames}, 实际时长: {actual_duration:.2f}s")

        import requests

        create_url = "https://apihub.agnes-ai.com/v1/videos"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
            raise RuntimeError(f"创建任务失败: {str(e)}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有video_id: {task_data}")

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
        query_url = f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}&model_name=agnes-video-v2.0"
        headers = {"Authorization": f"Bearer {api_key}"}

        from comfy.utils import ProgressBar
        pbar = ProgressBar(max_retries)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)
                pbar.update_absolute(attempt, max_retries)

                display_status = status
                if status == "queued":
                    display_status = "排队中"
                elif status == "in_progress":
                    display_status = "视频生成中"
                elif status == "processing":
                    display_status = "处理中"
                elif status == "completed":
                    display_status = "已完成"
                elif status == "failed":
                    display_status = "失败"

                print(f"[Agnes] 状态: {display_status}, 进度: {progress}%")

                if progress == last_progress and status in ["in_progress", "processing"]:
                    stuck_count += 1
                    if stuck_count >= 10:
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed":
                    video_url = result.get("remixed_from_video_id")
                    if not video_url:
                        video_url = result.get("video_url") or result.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL: {result}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败: {error_msg}")
                elif status in ["queued", "in_progress", "processing"]:
                    time.sleep(interval)
                else:
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错: {e}, 重试中...")
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
                    raise RuntimeError(f"下载视频失败: {str(e)}")

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
                "image1": ("IMAGE", {"tooltip": "第一张参考图片（必选）"}),
                "mode": (["multi-image", "keyframes"], {
                    "default": "multi-image",
                    "tooltip": "生成模式：multi-image=多图引导生成，keyframes=关键帧之间平滑过渡"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Create a smooth transformation between the reference images, maintaining visual consistency and natural motion",
                    "tooltip": "描述视频生成内容，尤其描述图片之间的过渡或动作"
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
                "image2": ("IMAGE", {"tooltip": "第二张参考图片（可选）"}),
                "image3": ("IMAGE", {"tooltip": "第三张参考图片（可选）"}),
                "image4": ("IMAGE", {"tooltip": "第四张参考图片（可选）"}),
                "image5": ("IMAGE", {"tooltip": "第五张参考图片（可选）"}),
                "image6": ("IMAGE", {"tooltip": "第六张参考图片（可选）"}),
                "image7": ("IMAGE", {"tooltip": "第七张参考图片（可选）"}),
                "image8": ("IMAGE", {"tooltip": "第八张参考图片（可选）"}),
                "image9": ("IMAGE", {"tooltip": "第九张参考图片（可选）"}),
                "image10": ("IMAGE", {"tooltip": "第十张参考图片（可选）"}),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "generate_video"
    CATEGORY = "智绘Store/Agens AI"
    DESCRIPTION = "使用Agnes-Video-V2.0根据多张参考图片生成视频。支持多图引导生成或关键帧动画。第1张图片必填，其余可选。"

    def generate_video(self, image1: torch.Tensor, mode: str, prompt: str, negative_prompt: str,
                       width: int, height: int, frame_rate: float, duration_seconds: float,
                       seed: int, num_inference_steps: int = 50,
                       image2=None, image3=None, image4=None, image5=None,
                       image6=None, image7=None, image8=None, image9=None, image10=None) -> Tuple[torch.Tensor, Optional[Dict], float]:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("未找到Agnes API Key，请在ComfyUI设置中配置（Agnes AI API Key）")

        # 收集所有非空图像，顺序为 image1, image2, ..., image10
        images = [image1]
        for img in [image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if img is not None:
                images.append(img)

        if len(images) < 2 and mode == "keyframes":
            print("[Agnes] 关键帧模式至少需要2张图片，当前只有1张，将使用多图模式")
            mode = "multi-image"

        # 将图像列表转换为 Base64 字符串列表
        image_base64_list = [self._image_to_base64(img) for img in images]
        print(f"[Agnes] 已转换 {len(image_base64_list)} 张图片为 Base64")

        num_frames = duration_to_frames(duration_seconds, frame_rate)
        actual_duration = num_frames / frame_rate
        print(f"[Agnes] 目标时长: {duration_seconds}s, 帧率: {frame_rate}fps -> 使用帧数: {num_frames}, 实际时长: {actual_duration:.2f}s")

        import requests

        create_url = "https://apihub.agnes-ai.com/v1/videos"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # 构建 extra_body
        extra_body = {"image": image_base64_list}
        if mode == "keyframes":
            extra_body["mode"] = "keyframes"

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
            "seed":seed
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
            raise RuntimeError(f"创建任务失败: {str(e)}")

        video_id = task_data.get("video_id")
        if not video_id:
            raise RuntimeError(f"创建任务响应中没有video_id: {task_data}")

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
        """将 ComfyUI 图像 Tensor 转换为纯 Base64 字符串（无 data URL 前缀）"""
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
        query_url = f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}&model_name=agnes-video-v2.0"
        headers = {"Authorization": f"Bearer {api_key}"}

        from comfy.utils import ProgressBar
        pbar = ProgressBar(max_retries)

        last_progress = -1
        stuck_count = 0

        for attempt in range(max_retries):
            try:
                resp = requests.get(query_url, headers=headers, timeout=(5, 15))
                resp.raise_for_status()
                result = resp.json()
                status = result.get("status")
                progress = result.get("progress", 0)
                pbar.update_absolute(attempt, max_retries)

                display_status = status
                if status == "queued":
                    display_status = "排队中"
                elif status == "in_progress":
                    display_status = "视频生成中"
                elif status == "processing":
                    display_status = "处理中"
                elif status == "completed":
                    display_status = "已完成"
                elif status == "failed":
                    display_status = "失败"

                print(f"[Agnes] 状态: {display_status}, 进度: {progress}%")

                if progress == last_progress and status in ["in_progress", "processing"]:
                    stuck_count += 1
                    if stuck_count >= 10:
                        print(f"[Agnes] 警告: 进度停滞在 {progress}% 超过10次，继续等待...")
                        stuck_count = 0
                else:
                    last_progress = progress
                    stuck_count = 0

                if status == "completed":
                    video_url = result.get("remixed_from_video_id")
                    if not video_url:
                        video_url = result.get("video_url") or result.get("url")
                    if not video_url:
                        raise RuntimeError(f"任务完成但未找到视频URL: {result}")
                    return video_url
                elif status == "failed":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"视频生成失败: {error_msg}")
                elif status in ["queued", "in_progress", "processing"]:
                    time.sleep(interval)
                else:
                    time.sleep(interval)
            except requests.exceptions.RequestException as e:
                print(f"[Agnes] 查询出错: {e}, 重试中...")
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
                    raise RuntimeError(f"下载视频失败: {str(e)}")

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
        url = "https://apihub.agnes-ai.com/v1/images/generations"
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
            raise RuntimeError(f"图像生成请求失败: {str(e)}")

        # 解析Base64图片数据
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"响应中没有data字段: {result}")
        b64_json = data[0].get("b64_json")
        if not b64_json:
            # 降级：尝试url输出
            img_url = data[0].get("url")
            if img_url:
                print(f"[Agnes] 响应为URL，将下载图片: {img_url}")
                img_data = self._download_image(img_url)
            else:
                raise RuntimeError(f"响应中没有图片数据: {result}")
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
        url = "https://apihub.agnes-ai.com/v1/images/generations"
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
            raise RuntimeError(f"图生图请求失败: {str(e)}")

        # 解析Base64图片数据
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"响应中没有data字段: {result}")
        b64_json = data[0].get("b64_json")
        if not b64_json:
            # 降级：尝试url
            img_url = data[0].get("url")
            if img_url:
                print(f"[Agnes] 响应为URL，将下载图片: {img_url}")
                img_data = self._download_image(img_url)
            else:
                raise RuntimeError(f"响应中没有图片数据: {result}")
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
        # url = "https://apihub.agnes-ai.com/v1/images/generations"
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

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AgnesTextToVideo": AgnesTextToVideo,
    "AgnesImageToVideo": AgnesImageToVideo,
    "AgnesMultiImageToVideo": AgnesMultiImageToVideo, 
    "AgnesTextToImage": AgnesTextToImage,
    "AgnesImageToImage": AgnesImageToImage,
    # "AgnesMultiImageToImage": AgnesMultiImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgnesTextToVideo": "Agnes 文生视频",
    "AgnesImageToVideo": "Agnes 单图生视频",
    "AgnesMultiImageToVideo": "Agnes 多图生视频",
    "AgnesTextToImage": "Agnes 文生图",
    "AgnesImageToImage": "Agnes 图生图",
    # "AgnesMultiImageToImage": "Agnes 多图编辑",
}