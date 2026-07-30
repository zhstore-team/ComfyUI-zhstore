import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Agnes.AgnesChat",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AgnesChat" && nodeData.name !== "AgnesVisionChat") return;

        const isVision = nodeData.name === "AgnesVisionChat";
        const propKey = isVision ? "_agnes_chat_history_vision" : "_agnes_chat_history";

        // ---- onNodeCreated ----
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            if (this.widgets) {
                const promptWidget = this.widgets.find(w => w.name === "user_prompt");
                if (promptWidget && promptWidget.inputEl) {
                    promptWidget.inputEl.placeholder = "在此输入您的问题...";
                }
            }

            return result;
        };

        // ---- onExecuted：执行完毕后将历史保存到 properties（随工作流 JSON 持久化）----
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.history) {
                this.properties = this.properties || {};
                this.properties[propKey] = message.history;
            }
        };
    },
});
