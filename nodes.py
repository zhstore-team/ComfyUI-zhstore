import torch
import numpy as np
import base64
import re
from PIL import Image
import io

class BooleanSwitch:
    """
    根据布尔开关选择输出：
    - 当开关为 True 时，输出“真”输入端的数据
    - 当开关为 False 时，输出“假”输入端的数据
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "布尔": ("BOOLEAN", {"default": True, "label": "布尔"}),
            },
            "optional": {
                "真": ("*",),   # 任意类型，未连线时默认为 None
                "假": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("输出",)
    FUNCTION = "switch_output"
    CATEGORY = "智绘Store/判断"

    def switch_output(self, 布尔, 真=None, 假=None):
        if 布尔:
            return (真,)
        else:
            return (假,)
            
 
class TextCombiner:
    """
    将最多5个文本参数拼接成一个字符串，自动忽略空值。
    - 未开启换行：使用指定的分隔符拼接（分隔符仅出现在段落之间）。
    - 开启换行：每个非空文本后面都加上【分隔符+换行符】（包括最后一个文本）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本1": ("STRING", {"default": "", "multiline": True}),
                "文本2": ("STRING", {"default": "", "multiline": True}),
                "文本3": ("STRING", {"default": "", "multiline": True}),
                "文本4": ("STRING", {"default": "", "multiline": True}),
                "文本5": ("STRING", {"default": "", "multiline": True}),
                "分隔符": ("STRING", {"default": "", "multiline": False}),
                "换行": ("BOOLEAN", {"default": False, "label": "每段后加分隔符+换行（含末行）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "combine"
    CATEGORY = "智绘Store/文本工具"

    def combine(self, 文本1, 文本2, 文本3, 文本4, 文本5, 分隔符, 换行):
        # 收集非空文本（去除首尾空格后不为空）
        parts = []
        for text in (文本1, 文本2, 文本3, 文本4, 文本5):
            if text and text.strip() != "":
                parts.append(text)

        if not parts:
            return ("",)

        if 换行:
            # 判断分隔符是否有效（非空且非纯空格）
            sep = 分隔符 if (分隔符 and 分隔符.strip() != "") else ""
            # 每个文本后都加上分隔符和换行符（包括最后一个）
            result = "".join([text + sep + "\n" for text in parts])
        else:
            # 未开启换行：使用用户指定的分隔符（无效则直接拼接）
            sep = 分隔符 if (分隔符 and 分隔符.strip() != "") else ""
            result = sep.join(parts)

        return (result,)

class TextCombiner10:
    """
    将最多10个文本参数拼接成一个字符串，自动忽略空值。
    - 未开启换行：使用指定的分隔符拼接（分隔符仅出现在段落之间）。
    - 开启换行：每个非空文本后面都加上【分隔符+换行符】（包括最后一个文本）。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本1": ("STRING", {"default": "", "multiline": True}),
                "文本2": ("STRING", {"default": "", "multiline": True}),
                "文本3": ("STRING", {"default": "", "multiline": True}),
                "文本4": ("STRING", {"default": "", "multiline": True}),
                "文本5": ("STRING", {"default": "", "multiline": True}),
                "文本6": ("STRING", {"default": "", "multiline": True}),
                "文本7": ("STRING", {"default": "", "multiline": True}),
                "文本8": ("STRING", {"default": "", "multiline": True}),
                "文本9": ("STRING", {"default": "", "multiline": True}),
                "文本10": ("STRING", {"default": "", "multiline": True}),
                "分隔符": ("STRING", {"default": "", "multiline": False}),
                "换行": ("BOOLEAN", {"default": False, "label": "每段后加分隔符+换行（含末行）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "combine"
    CATEGORY = "智绘Store/文本工具"

    def combine(self, 文本1, 文本2, 文本3, 文本4, 文本5, 文本6, 文本7, 文本8, 文本9, 文本10, 分隔符, 换行):
        # 收集非空文本（去除首尾空格后不为空）
        parts = []
        for text in (文本1, 文本2, 文本3, 文本4, 文本5, 文本6, 文本7, 文本8, 文本9, 文本10):
            if text and text.strip() != "":
                parts.append(text)

        if not parts:
            return ("",)

        if 换行:
            # 判断分隔符是否有效（非空且非纯空格）
            sep = 分隔符 if (分隔符 and 分隔符.strip() != "") else ""
            # 每个文本后都加上分隔符和换行符（包括最后一个）
            result = "".join([text + sep + "\n" for text in parts])
        else:
            # 未开启换行：使用用户指定的分隔符（无效则直接拼接）
            sep = 分隔符 if (分隔符 and 分隔符.strip() != "") else ""
            result = sep.join(parts)

        return (result,)

        

# ========== 新增节点：GroupByenable（参数汉化） ==========
class GroupByenable:
    """
    控制所在分组内所有节点的启用/旁路状态。
    实际逻辑由前端 JavaScript 完成。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "启用分组": ("BOOLEAN", {
                    "default": True,
                    "label": "启用分组"
                }),
            },
        }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "toggle_group"
    CATEGORY = "智绘Store/分组控制"

    def toggle_group(self, 启用分组):
        # 后端不需要做任何事，前端会监听这个参数的变化并执行分组控制
        return ()


# ========== 新增节点：GroupBypass（参数汉化） ==========
class GroupBypass:
    """
    控制所在分组内所有节点进入 bypass 状态（紫色）。
    实际逻辑由前端 JavaScript 完成。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "绕过分组": ("BOOLEAN", {
                    "default": True,
                    "label": "绕过分组 (Bypass Group)"
                }),
            },
        }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "toggle_group_bypass"
    CATEGORY = "智绘Store/分组控制"

    def toggle_group_bypass(self, 绕过分组):
        # 后端无需操作，前端监听 widget 变化
        return ()


# ========== 尺寸预选节点（参数汉化） ==========
class ImageResolutionPreset:
    """
    尺寸预选节点：通过下拉菜单选择预设分辨率，输出宽度和高度。
    """
    PRESETS = {
        # 1:1
        "480 x 480 【1:1】": (480, 480),
        "512 x 512 【1:1】": (512, 512),
        "768 x 768 【1:1】": (768, 768),
        "1024 x 1024 【1:1】": (1024, 1024),
        "1536 x 1536 【1:1】": (1536, 1536),
        "2048 x 2048 【1:1】": (2048, 2048),
        "4096 x 4096 【1:1】": (4096, 4096),
        # 4:3 及 3:4
        "480 x 640 【3:4】": (480, 640),
        "768 x 1024 【3:4】": (768, 1024),
        "960 x 1280 【3:4】": (960, 1280),
        "1200 x 1600 【3:4】": (1200, 1600),
        "1440 x 1920 【3:4】": (1440, 1920),
        "1536 x 2048 【3:4】": (1536, 2048),
        "1920 x 2560 【3:4】": (1920, 2560),
        
        "640 x 480 【4:3】": (640, 480),
        "1024 x 768 【4:3】": (1024, 768),
        "1280 x 960 【4:3】": (1280, 960),
        "1600 x 1200 【4:3】": (1600, 1200),
        "1920 x 1440 【4:3】": (1920, 1440),        
        "2048 x 1536 【4:3】": (2048, 1536),
        "2560 x 1920 【4:3】": (2560, 1920),
        
        # 3:2 及 2:3
        "480 x 720 【2:3】": (480, 720),
        "832 x 1216 【2:3】": (832, 1216),
        "960 x 1440 【2:3】": (960, 1440),
        "1024 x 1536 【2:3】": (1024, 1536),       
        "1200 x 1800 【2:3】": (1200, 1800),
        "1280 x 1920 【2:3】": (1280, 1920),
        
        "720 x 480 【3:2】": (720, 480),
        "1216 x 832 【3:2】": (1216, 832),
        "1440 x 960 【3:2】": (1440, 960),
        "1536 x 1024 【3:2】": (1536, 1024),
        "1600 x 2400 【2:3】": (1600, 2400),
        "1800 x 1200 【3:2】": (1800, 1200),
        "1920 x 1280 【3:2】": (1920, 1280),
        "2400 x 1600 【3:2】": (2400, 1600),
        # 7:4 及 4:7
        "480 x 840 【4:7】": (480, 840),
        "800 x 1400 【4:7】": (800, 1400),
        "1024 x 1792 【4:7】": (1024, 1792),
        "1280 x 2240 【4:7】": (1280, 2240),
        
        "840 x 480 【7:4】": (840, 480),
        "1400 x 800 【7:4】": (1400, 800),
        "1792 x 1024 【7:4】": (1792, 1024),
        "2240 x 1280 【7:4】": (2240, 1280),
        
        
        # 16:9 及 9:16
        "480 x 854 【9:16】": (480, 854),
        "720 x 1280 【9:16】": (720, 1280),
        "768 x 1344 【9:16】": (768, 1344),
        "1080 x 1920 【9:16】": (1080, 1920),
        "1280 x 2304 【9:16】": (1280, 2304),
        "1440 x 2560 【9:16】": (1440, 2560),
        "2160 x 3840 【9:16】": (2160, 3840),
        
        "854 x 480 【16:9】": (854, 480),
        "1280 x 720 【16:9】": (1280, 720),
        "1344 x 768 【16:9】": (1344, 768),
        "1920 x 1080 【16:9】": (1920, 1080),       
        "2304 x 1280 【16:9】": (2304, 1280),
        "2560 x 1440 【16:9】": (2560, 1440),
        "3840 x 2160 【16:9】": (3840, 2160),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "预设尺寸": (list(cls.PRESETS.keys()), {
                    "default": "1024 x 1024 【1:1】"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "智绘Store/预设选择器"

    def get_resolution(self, 预设尺寸):
        width, height = self.PRESETS[预设尺寸]
        return (width, height)


#判断输入图像是竖版还是横版并输出对应尺寸的ComfyUI节点
class ImageOrientationDetector:
    """
    判断输入图像是竖版还是横版并输出对应尺寸的ComfyUI节点（完全汉化版）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "宽高比阈值": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "判断图像方向的阈值，默认保持1就可以了"
                }),
                "方形宽度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【方形】时，宽度端口输出这里的参数"
                }),
                "方形高度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【方形】时，高度端口输出这里的参数"
                }),
                "横版宽度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【横版】时，宽度端口输出这里的参数"
                }),
                "横版高度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【横版】时，高度端口输出这里的参数"
                }),
                "竖版宽度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【竖版】时，宽度端口输出这里的参数"
                }),
                "竖版高度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1,
                    "tooltip": "当图像为【竖版】时，高度端口输出这里的参数"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("方向", "是否横版", "宽度", "高度")
    FUNCTION = "detect_orientation"
    CATEGORY = "智绘Store/图像工具"
    OUTPUT_NODE = True

    def detect_orientation(self, 图像, 宽高比阈值=1.0,
                          方形宽度=0, 方形高度=0,
                          横版宽度=0, 横版高度=0,
                          竖版宽度=0, 竖版高度=0):
        """
        检测图像方向并返回对应的目标尺寸
        """
        # 处理批次维度
        if len(图像.shape) == 4:
            图像 = 图像[0]
        
        # 将张量转换为 numpy 数组
        image_np = 图像.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 获取图像尺寸
        if len(image_np.shape) == 3:
            h, w = image_np.shape[:2]
        else:
            h, w = image_np.shape[-2:]
        
        # 计算宽高比
        aspect_ratio = w / h
        
        # 判断方向
        is_landscape = aspect_ratio > 宽高比阈值
        
        if is_landscape:
            orientation = "横版"
            target_width = 横版宽度
            target_height = 横版高度
        elif aspect_ratio < 1.0 / 宽高比阈值:
            orientation = "竖版"
            target_width = 竖版宽度
            target_height = 竖版高度
        else:
            orientation = "方形"
            target_width = 方形宽度
            target_height = 方形高度
        
        # print(f"图像尺寸: {w}x{h}, 宽高比: {aspect_ratio:.3f}, 方向: {orientation}")
        # print(f"宽高输出: {target_width}x{target_height}")
        
        return (orientation, is_landscape, target_width, target_height)


class ValueEqualityChecker:
    """
    比较任意输入值与预设字符串是否相等，输出布尔值。
    """
    DESCRIPTION = "一个强大的值比较器，用于判断输入与预设值是否匹配，匹配输出true，不匹配输出false"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "比较值": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "label": "比较值"
                }),
            },
            "optional": {
                "任何输入": ("*",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("布尔",)
    FUNCTION = "check_equality"
    CATEGORY = "智绘Store/判断"
    OUTPUT_NODE = False

    def check_equality(self, 比较值, 任何输入=None):
        # 将任意输入转换为字符串进行比较
        if 任何输入 is None:
            input_str = ""
        elif isinstance(任何输入, bool):
            # 布尔值转换为小写字符串 "true" 或 "false"
            input_str = str(任何输入).lower()
        else:
            input_str = str(任何输入)
        
        # 比较
        result = (input_str == 比较值)
        return (result,)



import torch
import numpy as np
import cv2
import base64
from PIL import Image
import folder_paths
import uuid
import os

class DecodeBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_data": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # 关键修复点1：加上逗号，使其成为包含一个元素的元组
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True   # 作为输出节点，允许预览
    FUNCTION = "load_image"
    CATEGORY = "智绘Store/图像工具"

    def convert_color(self, image):
        # 将BGRA或BGR转换为RGB
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_image(self, base64_data, prompt=None, extra_pnginfo=None):
        # 默认空张量
        empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # 无效输入处理
        if not base64_data or base64_data.strip() == "":
            return {"ui": {"images": []}, "result": (empty_img,)}

        # 解码 base64
        try:
            nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
        except Exception as e:
            print(f"[Base64ToImage] Base64解码失败: {e}")
            return {"ui": {"images": []}, "result": (empty_img,)}

        # 使用 cv2 解码图像
        result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if result is None:
            print("[Base64ToImage] cv2.imdecode 失败，可能不是有效的图像数据")
            return {"ui": {"images": []}, "result": (empty_img,)}

        # 转换颜色空间并归一化
        result_rgb = self.convert_color(result)
        result_norm = result_rgb.astype(np.float32) / 255.0
        new_images = torch.from_numpy(result_norm)[None, :, :, :]   # [1, H, W, C]

        # ---------- 生成自身预览（临时文件方式）----------
        preview_images = []
        try:
            temp_dir = folder_paths.get_temp_directory()
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            unique_name = f"preview_{uuid.uuid4().hex}.png"
            temp_file = os.path.join(temp_dir, unique_name)
            Image.fromarray(result_rgb).save(temp_file, format="PNG")
            preview_images = [{"filename": unique_name, "subfolder": "", "type": "temp"}]
        except Exception as e:
            print(f"[Base64ToImage] 生成预览失败: {e}")

        # 关键修复点2：返回正确的字典格式，result 必须是元组
        return {"ui": {"images": preview_images}, "result": (new_images,)}
 

import torch
import numpy as np
import base64
from PIL import Image
import io

class EncodeBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),   # 输入图像张量 [B, H, W, C], 范围 0~1
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"
    CATEGORY = "智绘Store/图像工具"
    DESCRIPTION = "将输入图像编码为 Base64 字符串（PNG 格式）"

    def encode(self, image):
        # 取批次中的第一张图像
        img_tensor = image[0]           # [H, W, C]
        img_np = img_tensor.cpu().numpy()  # 转为 numpy 数组
        img_np = (img_np * 255).astype(np.uint8)  # 从 0~1 映射到 0~255

        # 转为 PIL Image
        pil_img = Image.fromarray(img_np, mode='RGB')

        # 编码为 PNG 字节流，再转 Base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return (b64_str,)


import torch
import numpy as np
import base64
from PIL import Image
import io
import folder_paths
import os

class LoadImageToBase64:
    @classmethod
    def INPUT_TYPES(s):
        # 获取 ComfyUI 输入目录下的所有图像文件
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                files.append(f)
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "enable_resize": ("BOOLEAN", {"default": False, "label": "启用尺寸处理"}),
                "max_dimension": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1, "label": "最大边长"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_and_encode"
    CATEGORY = "智绘Store/图像工具"
    OUTPUT_NODE = True
    DESCRIPTION = "加载本地图像，可限制最大边长，输出 PNG 格式的 Base64 编码字符串"

    def load_and_encode(self, image, enable_resize, max_dimension, prompt=None, extra_pnginfo=None):
        # 构建完整文件路径
        image_path = folder_paths.get_annotated_filepath(image)

        # 打开图像并转为 RGB
        img = Image.open(image_path).convert('RGB')

        # 如果启用尺寸处理，调整图像大小（保持宽高比）
        if enable_resize:
            original_width, original_height = img.size
            max_side = max(original_width, original_height)
            if max_side > max_dimension:
                scale = max_dimension / max_side
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                # 使用高质量重采样
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 编码为 PNG 字节流
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 可选：生成预览图像（展示处理后/原图）
        # 为了简洁，这里只输出字符串；如需预览可取消下方注释
        # preview_images = []
        # try:
        #     temp_dir = folder_paths.get_temp_directory()
        #     if not os.path.exists(temp_dir):
        #         os.makedirs(temp_dir)
        #     unique_name = f"preview_{uuid.uuid4().hex}.png"
        #     temp_file = os.path.join(temp_dir, unique_name)
        #     img.save(temp_file, format="PNG")
        #     preview_images = [{"filename": unique_name, "subfolder": "", "type": "temp"}]
        #     return {"ui": {"images": preview_images}, "result": (b64_str,)}
        # except Exception as e:
        #     print(f"[LoadImageToBase64] 生成预览失败: {e}")

        return (b64_str,)

    # 让 ComfyUI 能够从输入目录中动态刷新文件列表
    @classmethod
    def IS_CHANGED(s, image, enable_resize, max_dimension):
        image_path = folder_paths.get_annotated_filepath(image)
        return os.path.getmtime(image_path)      

class ShowTextByPerpetual:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()          # 空元组表示无输出端口
    FUNCTION = "notify"
    OUTPUT_NODE = True         # 仍是输出节点，可更新界面
    CATEGORY = "智绘Store/文本工具"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        # 只返回 UI 更新信息，不返回 result（因此没有输出端口）
        return {"ui": {"text": text}}
    



  
# ========== 节点注册 ==========
NODE_CLASS_MAPPINGS = {
    "GroupByenable": GroupByenable,
    "GroupBypass": GroupBypass,
    "ImageResolutionPreset": ImageResolutionPreset,
    "ImageOrientationDetector": ImageOrientationDetector,
    "BooleanSwitch": BooleanSwitch,
    "ValueEqualityChecker": ValueEqualityChecker,
    "TextCombiner": TextCombiner,
    "TextCombiner10": TextCombiner10,
    "DecodeBase64": DecodeBase64,
    "EncodeBase64": EncodeBase64,
    "LoadImageToBase64": LoadImageToBase64,
    "ShowTextByPerpetual": ShowTextByPerpetual,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroupByenable": "分组启用开关",
    "GroupBypass": "分组绕过开关",
    "ImageResolutionPreset": "尺寸预选节点",
    "ImageOrientationDetector": "图像方向检测器",
    "BooleanSwitch": "布尔选择输出",
    "ValueEqualityChecker": "输入比较器",
    "TextCombiner": "文本组合",
    "TextCombiner10": "文本组合（10行文本）",
    "DecodeBase64": "Base64 解码",
    "EncodeBase64": "Base64 编码（图像转）",
    "LoadImageToBase64": "Base64 编码（加载图像）",
    "ShowTextByPerpetual": "显示文本（持久化）",

}