# 关于我们
# 智绘Store+AI库：在线学习资料库
**智绘Store+AI库**是由智绘Store团队开发的一款微信小程序，专注分享ComfyUI及AI相关的知识与服务，提供在线学习资料：ComfyUI教程、节点手册、模型教程、工作流分享、行业资讯等。

<img width="800" height="400" alt="xcx_1" src="https://github.com/user-attachments/assets/57701b89-fbc1-4287-8cf9-9118ea8c0eaa" />
<br><br>
<img width="800" height="400" alt="gzh" src="https://github.com/user-attachments/assets/e126d9ea-5493-4580-8b7f-c980bf078965" />
<br><br>

## 更新说明

- 由于 Agnes 官方 API 调整，原 **Agnes 多图生视频**节点已升级为 **Agnes 首尾帧生视频**节点。新节点支持首帧、中帧（可选）、尾帧参考图，用于生成首尾帧之间的平滑过渡动画。

## 安装教程

1. 进入 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
```

2. 克隆本仓库：
```bash
git clone https://github.com/zhstore-comfyui/ComfyUI-zhstore.git
```

3. 安装依赖（**必须步骤**）：
```bash
cd ComfyUI-zhstore
pip install -r requirements.txt
```

4. 重启 ComfyUI，插件即可生效。

> **注意**：Agnes AI 相关节点（文生图、图生图、文生视频、单图生视频、首尾帧生视频）需要在 ComfyUI 设置中配置 Agnes API Key 才能使用。


## 插件介绍
ComfyUI-zhstore插件包，是智绘Store团队开发的一款适用于ComfyUI的节点插件包，提供了多种类型的插件，主要涵盖了以下节点：
### 1、判断
- 输入比较器

- 布尔选择输出

### 2、分组控制
- 分组启用开关

- 分组绕过开关

### 3、预设选择器
- 尺寸预选节点

### 4、图像工具
- 图像方向检测器

### 5、文本工具
- 文本组合

显示文本（持久化）

### 6、Agnes AI
- Agnes 文生图

- Agnes 图生图

- Agnes 文生视频

- Agnes 单图生视频

- Agnes 首尾帧生视频
<br><br>

## 示例

> #### 分组绕过开关
> 适用于在子图节点上控制子图内节点组框是否bypass，只需要将 **分组绕过开关** 放置到需要控制的组框内即可。
<img width="3370" height="805" alt="pass_1" src="https://github.com/user-attachments/assets/f7f38307-3e98-4521-bbbe-e0f1568d1eb9" />

<br><br>
> #### Agnes 单图编辑

<img width="2246" height="693" alt="i2i_1" src="https://github.com/user-attachments/assets/68f6a91e-b428-4491-a093-5ab414e134f5" />
<br><br>

> #### Agnes 单图生视频

<img width="1297" height="505" alt="t2v_1" src="https://github.com/user-attachments/assets/bf535d5a-4517-47e0-9899-e20c93bf0791" />



