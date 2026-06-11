import { app } from "../../../scripts/app.js";

/**
 * 获取节点所在的分组（Group）
 */
function getNodeGroup(node) {
    if (!node.graph) return null;
    for (const group of node.graph._groups) {
        if (node.pos[0] >= group.pos[0] &&
            node.pos[0] + node.size[0] <= group.pos[0] + group.size[0] &&
            node.pos[1] >= group.pos[1] &&
            node.pos[1] + node.size[1] <= group.pos[1] + group.size[1]) {
            return group;
        }
    }
    return null;
}

/**
 * 获取指定分组内的所有节点（包括嵌套分组中的节点）
 * 基于坐标包含关系一次性遍历所有节点，无需递归
 */
function getAllNodesInGroup(group) {
    if (!group || !group.graph) return [];
    const nodes = [];
    for (const node of group.graph._nodes) {
        if (node.pos[0] >= group.pos[0] &&
            node.pos[0] + node.size[0] <= group.pos[0] + group.size[0] &&
            node.pos[1] >= group.pos[1] &&
            node.pos[1] + node.size[1] <= group.pos[1] + group.size[1]) {
            nodes.push(node);
        }
    }
	console.log(nodes)
    return nodes;
}

/**
 * 更新分组内所有节点的 bypass 状态
 * enabled = true  → 正常执行 (mode = 0)
 * enabled = false → bypass   (mode = 2)
 */
function updateGroupNodesMode(group, enabled) {
    const targetMode = enabled ? 0 : 2;
    const nodes = getAllNodesInGroup(group);
    for (const node of nodes) {
        if (node.mode !== targetMode) {
            node.mode = targetMode;
            // 刷新节点显示样式
            if (node.onModeChange) node.onModeChange?.(targetMode);
            if (node.updateBypassedStyle) node.updateBypassedStyle?.();
        }
    }
    group.graph.setDirtyCanvas(true, true);
}

/**
 * 注册扩展：监听 GroupByenable 节点的创建和移动
 */
function initGroupToggle() {
    app.registerExtension({
        name: "Comfy.GroupByenable",
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== "GroupByenable") return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                if (this.widgets) {
                    const enabledWidget = this.widgets.find(w => w.name === "启用分组");
                    if (enabledWidget) {
                        // 监听开关值变化
                        enabledWidget.callback = (value) => {
                            const group = getNodeGroup(this);
                            if (group) updateGroupNodesMode(group, value);
                        };
                        // 初始化时执行一次
                        const group = getNodeGroup(this);
                        if (group) updateGroupNodesMode(group, enabledWidget.value);
                    }
                }
                return result;
            };

            const onNodeMoved = nodeType.prototype.onNodeMoved;
            nodeType.prototype.onNodeMoved = function() {
                const result = onNodeMoved?.apply(this, arguments);
                if (this.widgets) {
                    const enabledWidget = this.widgets.find(w => w.name === "启用分组");
                    if (enabledWidget) {
                        const group = getNodeGroup(this);
                        if (group) updateGroupNodesMode(group, enabledWidget.value);
                    }
                }
                return result;
            };
        }
    });
}

initGroupToggle();