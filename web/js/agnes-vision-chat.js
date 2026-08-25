import { app } from "../../../scripts/app.js";

/**
 * Agnes 图像理解对话节点前端扩展
 * 图像端口自动衍生：接入某一张图像后，自动追加下一个可用的图像端口（图像1..N，上限 8）。
 * 所有图像端口均为可选输入，并统一排列在「系统提示词」端口上方。
 */

const NODE_NAME = "AgnesVisionChat";
const MAX_IMAGES = 8;

function syncImageInputs(node) {
    if (!node || !node.inputs) return;

    // 统计当前已连接的最高图像端口
    let maxConnected = 0;
    for (const inp of node.inputs) {
        if (inp && /^图像\d+$/.test(inp.name) && inp.link != null) {
            const n = parseInt(inp.name.slice(2), 10);
            if (n > maxConnected) maxConnected = n;
        }
    }

    // 目标端口数：保留「已连接最高端口 + 1」（始终留一个空端口便于连入下一张），至少 1 个
    const target = Math.min(MAX_IMAGES, Math.max(1, maxConnected + 1));

    // 1. 移除编号超出 target 的图像端口（均为未连接的空端口，自动断开无副作用）
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (inp && /^图像\d+$/.test(inp.name)) {
            const n = parseInt(inp.name.slice(2), 10);
            if (n > target) {
                node.removeInput(i);
            }
        }
    }

    // 2. 补齐缺失的图像端口（1..target），插入到「系统提示词」端口之前
    const existing = new Set(node.inputs.map((i) => i && i.name));
    const sysPromptIdx = node.inputs.findIndex((i) => i && i.name === "system_prompt");
    const insertAt = sysPromptIdx === -1 ? node.inputs.length : sysPromptIdx;

    for (let n = 1; n <= target; n++) {
        const name = "图像" + n;
        if (!existing.has(name)) {
            node.addInput(name, "IMAGE", { tooltip: "可选输入图像 " + n });
            // 将刚追加到末尾的端口移动到「系统提示词」上方
            const el = node.inputs.pop();
            node.inputs.splice(insertAt, 0, el);
        }
    }

    node.setSize(node.computeSize());
    if (node.graph) {
        node.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "Comfy.AgnesVisionChat",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // 连接/断开图像时自动衍生或复核端口
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function (...args) {
                const r = onConnectionsChange?.apply(this, args);
                setTimeout(() => syncImageInputs(this), 0);
                return r;
            };

            setTimeout(() => syncImageInputs(this), 0);
            return result;
        };
    },
});