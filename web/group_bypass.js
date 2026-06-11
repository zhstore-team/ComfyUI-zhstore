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
 * 获取指定分组内的所有节点（包括嵌套分组）
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
 * 更新分组内所有节点的模式
 * bypass = true  → 所有节点进入 bypass (mode = 2，紫色)
 * bypass = false → 恢复正常 (mode = 0)
 */
function updateGroupNodesMode(group, bypass) {
    const targetMode = bypass ? 0 : 4;
    const nodes = getAllNodesInGroup(group);
    for (const node of nodes) {
        if (node.mode !== targetMode) {
            node.mode = targetMode;
            if (node.onModeChange) node.onModeChange?.(targetMode);
            if (node.updateBypassedStyle) node.updateBypassedStyle?.();
        }
    }
    group.graph.setDirtyCanvas(true, true);
}

/**
 * 注册扩展：监听 GroupBypass 节点
 */
function initGroupBypass() {
    app.registerExtension({
        name: "Comfy.GroupBypass",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== "GroupBypass") return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                const bypassWidget = this.widgets?.find(w => w.name === "绕过分组");
                if (bypassWidget) {
                    bypassWidget.callback = (value) => {
                        const group = getNodeGroup(this);
                        if (group) updateGroupNodesMode(group, value);
                    };
                    const group = getNodeGroup(this);
                    if (group) updateGroupNodesMode(group, bypassWidget.value);
                }
                return result;
            };

            const onNodeMoved = nodeType.prototype.onNodeMoved;
            nodeType.prototype.onNodeMoved = function() {
                const result = onNodeMoved?.apply(this, arguments);
                const bypassWidget = this.widgets?.find(w => w.name === "绕过分组");
                if (bypassWidget) {
                    const group = getNodeGroup(this);
                    if (group) updateGroupNodesMode(group, bypassWidget.value);
                }
                return result;
            };
        }
    });
}

initGroupBypass();