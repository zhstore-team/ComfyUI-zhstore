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


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]