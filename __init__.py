"""
ComfyUI插件初始化文件
"""

import json
import os
import logging
from aiohttp import web
import server


from . import nodes          # 原有的节点模块（如果有）
from . import agnes_nodes    # 新增的 Agnes 模块


# 告诉 ComfyUI 前端资源所在的文件夹
WEB_DIRECTORY = "web"

# 合并节点映射
NODE_CLASS_MAPPINGS = {
    **getattr(nodes, "NODE_CLASS_MAPPINGS", {}),          # 原有节点
    **getattr(agnes_nodes, "NODE_CLASS_MAPPINGS", {}),    # 新增节点
}

# 可选：合并显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    **getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}),
    **getattr(agnes_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}),
}



# 配置日志
logger = logging.getLogger("AgnesAI")

# 配置文件路径
PLUGIN_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(PLUGIN_DIR, "agnes_config.json")

def load_agnes_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    return {}

def save_agnes_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

@server.PromptServer.instance.routes.post("/agnes/save_config")
async def save_config(request):
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        if not api_key:
            return web.json_response({"status": "error", "message": "API Key 不能为空"}, status=400)
        
        config = load_agnes_config()
        config["api_key"] = api_key
        if save_agnes_config(config):
            logger.info("API Key 已保存")
            return web.json_response({"status": "success", "message": "配置已保存"})
        else:
            return web.json_response({"status": "error", "message": "保存失败，请检查插件目录权限"}, status=500)
    except Exception as e:
        logger.error(f"保存配置时发生异常: {e}")
        return web.json_response({"status": "error", "message": f"服务器错误: {str(e)}"}, status=500)

@server.PromptServer.instance.routes.get("/agnes/get_config")
async def get_config(request):
    config = load_agnes_config()
    has_key = bool(config.get("api_key"))
    return web.json_response({"has_api_key": has_key})


# ====== 腾讯云 COS 配置（用于参考视频生成公网 URL）======
@server.PromptServer.instance.routes.post("/agnes/save_cos_config")
async def save_cos_config(request):
    try:
        data = await request.json()
        secret_id = (data.get("secret_id") or "").strip()
        secret_key = (data.get("secret_key") or "").strip()
        bucket = (data.get("bucket") or "").strip()
        region = (data.get("region") or "").strip()
        domain = (data.get("domain") or "").strip()

        config = load_agnes_config()
        config["cos_secret_id"] = secret_id
        config["cos_secret_key"] = secret_key
        config["cos_bucket"] = bucket
        config["cos_region"] = region
        config["cos_domain"] = domain
        if save_agnes_config(config):
            return web.json_response({"status": "success", "message": "COS 配置已保存"})
        else:
            return web.json_response({"status": "error", "message": "保存失败，请检查插件目录权限"}, status=500)
    except Exception as e:
        logger.error(f"保存 COS 配置时发生异常: {e}")
        return web.json_response({"status": "error", "message": f"服务器错误: {str(e)}"}, status=500)

@server.PromptServer.instance.routes.get("/agnes/get_cos_config")
async def get_cos_config(request):
    config = load_agnes_config()
    has_cos = bool(
        config.get("cos_secret_id")
        and config.get("cos_secret_key")
        and config.get("cos_bucket")
        and config.get("cos_region")
    )
    return web.json_response({
        "has_cos": has_cos,
        "cos_secret_id": config.get("cos_secret_id", ""),
        "cos_secret_key": config.get("cos_secret_key", ""),
        "cos_bucket": config.get("cos_bucket", ""),
        "cos_region": config.get("cos_region", ""),
        "cos_domain": config.get("cos_domain", ""),
    })


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]