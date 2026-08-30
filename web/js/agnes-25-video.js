import { app } from "../../../scripts/app.js";

/**
 * Agnes 2.5 全能视频节点前端扩展
 * 根据「模式」选择参数自动增删图像/音频输入端口，实现端口随模式切换。
 *
 * 模式与端口对应关系：
 * - 文生视频：无媒体端口
 * - 图生视频：首帧图
 * - 首尾帧  ：首帧图 + 尾帧图
 * - 全能参考：参考图1~5 + 参考音频1~3
 */

// 各模式应显示的媒体输入端口（顺序即端口顺序）
const MODE_INPUTS = {
    文生视频: [],
    图生视频: ["首帧图"],
    首尾帧: ["首帧图", "尾帧图"],
    全能参考: ["参考图1", "参考图2", "参考图3", "参考图4", "参考图5", "参考音频1", "参考音频2", "参考音频3"],
};

// 所有可被动态管理的媒体端口
const ALL_MEDIA_INPUTS = Object.values(MODE_INPUTS).flat().filter((v, i, a) => a.indexOf(v) === i);

// 端口名称 -> 类型
const INPUT_TYPES = {
    首帧图: "IMAGE",
    尾帧图: "IMAGE",
    参考图1: "IMAGE",
    参考图2: "IMAGE",
    参考图3: "IMAGE",
    参考图4: "IMAGE",
    参考图5: "IMAGE",
    参考音频1: "AUDIO",
    参考音频2: "AUDIO",
    参考音频3: "AUDIO",
};

function isDefaultSize(node) {
    // 判断节点当前尺寸是否仍是“默认尺寸”（即未被手动调整、也非复制/粘贴/载入工作流得到的尺寸）
    if (!node || !node.size || typeof node.computeSize !== "function") return true;
    const computed = node.computeSize();
    return Math.abs(node.size[0] - computed[0]) < 0.5 && Math.abs(node.size[1] - computed[1]) < 0.5;
}

function applyMode(node, mode) {
    if (!node || !node.inputs) return;

    const wanted = MODE_INPUTS[mode] || [];

    // 在改动端口前记录：节点是否为默认尺寸（复制/粘贴出的节点此时带有原节点的尺寸，不应被覆盖）
    const defaultSize = isDefaultSize(node);
    let changed = false;

    // 1. 移除当前模式不需要的媒体端口（removeInput 会自动断开连接）
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        // 仅移除「非当前模式需要 且 尚未连接」的媒体端口；
        // 已连接的端口一律保留，避免复制/粘贴时 ComfyUI 已恢复的连接被删而断线
        if (inp && ALL_MEDIA_INPUTS.includes(inp.name) && !wanted.includes(inp.name) && !inp.link) {
            node.removeInput(i);
            changed = true;
        }
    }

    // 2. 补上当前模式缺失的媒体端口（按 wanted 顺序追加）
    const existing = node.inputs.map((inp) => inp && inp.name);
    for (const name of wanted) {
        if (!existing.includes(name)) {
            node.addInput(name, INPUT_TYPES[name] || "IMAGE", { tooltip: "" });
            changed = true;
        }
    }

    // 3. 仅在“端口结构变化”且“节点仍为默认尺寸”时刷新布局，
    //    否则保留当前尺寸，避免复制/粘贴节点时尺寸被重置为默认大小
    if (changed && defaultSize) {
        node.setSize(node.computeSize());
    }
    if (node.graph) {
        node.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Comfy.Agnes25AllInOneVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Agnes25AllInOneVideo") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const modeWidget = this.widgets?.find((w) => w.name === "模式");
            if (modeWidget) {
                // 监听模式切换，端口随模式即时调整
                modeWidget.callback = (value) => applyMode(this, value);
                // 同步按当前模式应用一次端口（不使用 setTimeout，
                // 避免与复制/粘贴、载入工作流时的连接恢复产生竞态而断线）
                applyMode(this, modeWidget.value);
            }
            return result;
        };

        // 复制粘贴 / 载入工作流会走 configure 恢复端口与连接，
        // 在其返回后同步再对齐一次端口（applyMode 会保护已连接的端口）
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const modeWidget = this.widgets?.find((w) => w.name === "模式");
            if (modeWidget) {
                applyMode(this, modeWidget.value);
            }
            return result;
        };
    },
});