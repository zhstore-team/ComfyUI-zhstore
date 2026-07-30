/**
 * showMarkdown.js - Markdown 文档预览节点前端渲染
 * 使用 addDOMWidget 将渲染后的 HTML 作为自定义 DOM widget 注入节点。
 */
import { app } from "../../../scripts/app.js";

// 轻量级 Markdown → HTML 解析器
function parseMarkdown(src) {
  if (!src || !src.trim()) return "";
  var html = "";

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function inline(text) {
    text = text.replace(/`([^`]+)`/g, "<code style='background:rgba(255,255,255,0.08);color:#e06c75;padding:1px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:0.92em;'>$1</code>");
    text = text.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    text = text.replace(/___(.+?)___/g, "<strong><em>$1</em></strong>");
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__(.+?)__/g, "<strong>$1</strong>");
    text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
    text = text.replace(/_(.+?)_/g, "<em>$1</em>");
    text = text.replace(/~~(.+?)~~/g, "<del>$1</del>");
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, "<img src='$2' alt='$1' style='max-width:100%;border-radius:4px;'>");
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "<a href='$2' target='_blank' style='color:#58a6ff;'>$1</a>");
    return text;
  }

  var lines = src.split("\n");
  var i = 0;
  var inUl = false, inOl = false, inBq = false, inTable = false, inCode = false;
  var bqLines = [], codeLines = [], codeLang = "";

  function closeList() { if (inUl) { html += "</ul>\n"; inUl = false; } if (inOl) { html += "</ol>\n"; inOl = false; } }
  function closeBq() { if (inBq && bqLines.length) { html += "<blockquote style='margin:0.4em 0;padding:0.3em 0.8em;border-left:3px solid #5a5a5a;background:rgba(255,255,255,0.03);color:#b0b0b0;border-radius:0 4px 4px 0;'>" + inline(bqLines.join("<br>")) + "</blockquote>\n"; bqLines = []; inBq = false; } }
  function closeTable() { if (inTable) { html += "</tbody></table>\n"; inTable = false; } }

  while (i < lines.length) {
    var line = lines[i];

    if (!inCode && /^\s*```/.test(line)) { closeList(); closeBq(); closeTable(); inCode = true; codeLang = line.replace(/^\s*```/, "").trim(); codeLines = []; i++; continue; }
    if (inCode) {
      if (/^\s*```/.test(line)) { inCode = false; html += "<pre style='background:#1a1a2e;border:1px solid #2a2a3a;border-radius:6px;padding:8px 10px;margin:0.4em 0;overflow-x:auto;'><code" + (codeLang ? " class='language-" + esc(codeLang) + "'" : "") + " style='color:#c8d3f5;font-family:Consolas,monospace;font-size:0.88em;line-height:1.5;'>" + esc(codeLines.join("\n")) + "</code></pre>\n"; } else { codeLines.push(line); }
      i++; continue;
    }
    if (line.trim() === "") { closeList(); closeBq(); closeTable(); i++; continue; }
    if (/^\s*(?:---|\*\*\*|___)\s*$/.test(line)) { closeList(); closeBq(); closeTable(); html += "<hr style='border:none;height:1px;background:linear-gradient(to right,transparent,#555,transparent);margin:0.6em 0;'>\n"; i++; continue; }

    var hm = line.match(/^(#{1,6})\s+(.+)$/);
    if (hm) { closeList(); closeBq(); closeTable(); var lv = hm[1].length; html += "<h" + lv + " style='font-size:" + ({1:"1.5em",2:"1.3em",3:"1.15em",4:"1.05em",5:"1em",6:"0.95em"}[lv]||"1em") + ";font-weight:600;color:#f0f0f0;margin:0.5em 0 0.25em;" + (lv<=2?"border-bottom:1px solid #3a3a3a;padding-bottom:0.15em;":"") + "'>" + inline(hm[2]) + "</h" + lv + ">\n"; i++; continue; }

    if (/\|/.test(line) && i+1 < lines.length && /^\s*\|?\s*[-:]+[-|:\s]*$/.test(lines[i+1])) {
      closeList(); closeBq();
      var hdrs = line.replace(/^\||\|$/g,"").split("|").map(function(c){return c.trim();});
      var alns = lines[i+1].replace(/^\||\|$/g,"").split("|").map(function(c){var t=c.trim();return /^:-+:$/.test(t)?"center":/^-+:$/.test(t)?"right":"left";});
      html += "<table style='width:100%;border-collapse:collapse;margin:0.4em 0;font-size:0.92em;'><thead><tr>\n";
      hdrs.forEach(function(c,ci){html += "<th style='background:rgba(255,255,255,0.08);font-weight:600;color:#f0f0f0;padding:6px 8px;border:1px solid #3a3a3a;text-align:"+(alns[ci]||"left")+";'>"+inline(c)+"</th>\n";});
      html += "</tr></thead><tbody>\n"; inTable = true; i += 2;
      while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim() !== "") {
        var rcs = lines[i].replace(/^\||\|$/g,"").split("|").map(function(c){return c.trim();}); html += "<tr>";
        rcs.forEach(function(c,ci){html += "<td style='padding:5px 8px;border:1px solid #3a3a3a;color:#d0d0d0;'>"+inline(c)+"</td>";}); html += "</tr>\n"; i++;
      }
      closeTable(); continue;
    }

    var um = line.match(/^\s*[-*+]\s+(.+)$/);
    if (um) { closeBq(); closeTable(); if (!inUl) { closeList(); html += "<ul style='margin:0.3em 0;padding-left:1.5em;'>\n"; inUl = true; } html += "<li style='margin:0.1em 0;'>"+inline(um[1])+"</li>\n"; i++; continue; }

    var om = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (om) { closeBq(); closeTable(); if (!inOl) { closeList(); html += "<ol style='margin:0.3em 0;padding-left:1.5em;'>\n"; inOl = true; } html += "<li style='margin:0.1em 0;'>"+inline(om[1])+"</li>\n"; i++; continue; }

    var bm = line.match(/^\s*>\s?(.*)$/);
    if (bm) { closeList(); closeTable(); inBq = true; bqLines.push(bm[1]); i++; continue; }

    closeList(); closeBq(); closeTable();
    html += "<p style='margin:0.3em 0;color:#d4d4d4;'>"+inline(line)+"</p>\n"; i++;
  }

  if (inCode) { html += "<pre style='background:#1a1a2e;border:1px solid #2a2a3a;border-radius:6px;padding:8px 10px;margin:0.4em 0;overflow-x:auto;'><code style='color:#c8d3f5;font-family:Consolas,monospace;font-size:0.88em;'>"+esc(codeLines.join("\n"))+"</code></pre>\n"; }
  closeList(); closeBq(); closeTable();
  return html;
}

app.registerExtension({
  name: "zhstore.MarkdownViewer",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "MarkdownViewer") return;

    // 渲染 Markdown 内容到自定义 DOM widget
    function populate(mdText) {
      var self = this;
      // mdText 可能是字符串（onExecuted）或数组（onConfigure 传入 widgets_values.slice()）
      var text = Array.isArray(mdText) ? (mdText[0] || "") : (mdText || "");
      if (typeof text !== "string") text = String(text);

      var renderedHtml = parseMarkdown(text);

      // 查找已存在的渲染 div（由 onNodeCreated 创建），若不存在则创建
      var div = self._mdDiv;
      if (!div || !document.body.contains(div)) {
        div = document.createElement("div");
        div.style.cssText = [
          "padding:8px 10px",
          "overflow-y:auto",
          "overflow-x:hidden",
          "box-sizing:border-box",
          "color:#e0e0e0",
          "font-size:13px",
          "line-height:1.6",
          "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif",
          "word-wrap:break-word",
          "overflow-wrap:break-word",
          "word-break:break-word",
          "user-select:text",
          "-webkit-user-select:text",
          "border-radius:4px"
        ].join(";");
        self.addDOMWidget("md_html", "MARKDOWN_PREVIEW", div);
        self._mdDiv = div;

        // 滚动穿透控制
        div.addEventListener("wheel", function(e) {
          var atTop = div.scrollTop === 0;
          var atBottom = div.scrollTop + div.clientHeight >= div.scrollHeight - 1;
          if ((e.deltaY < 0 && atTop) || (e.deltaY > 0 && atBottom)) return;
          if (div.scrollHeight > div.clientHeight) {
            e.stopPropagation();
          }
        }, { passive: false });
      }

      // 更新内容
      div.innerHTML = renderedHtml || '<div style="color:#666;text-align:center;padding:60px 10px;">等待输入 Markdown 内容...</div>';

      // 内部图片、代码块和表格不应溢出容器
      var imgs = div.querySelectorAll("img, pre, table");
      for (var j = 0; j < imgs.length; j++) {
        imgs[j].style.maxWidth = "100%";
      }

      // 调整节点大小
      requestAnimationFrame(() => {
        const sz = self.computeSize();
        if (sz[0] < self.size[0]) sz[0] = self.size[0];
        if (sz[1] < self.size[1]) sz[1] = self.size[1];
        self.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
      });
    }

    // ------ onExecuted：执行完毕后显示内容 ------
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (message?.markdown) {
        var text = message.markdown[0] || "";
        populate.call(this, text);
        if (text) {
          this.properties = this.properties || {};
          this.properties.md_text = text;
        }
      }
    };

    // ------ 持久化：恢复（和 ShowText 完全一致）------
    const VALUES = Symbol();
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function () {
      this[VALUES] = arguments[0]?.widgets_values;
      var info = arguments[0];
      // 优先从 properties 恢复
      if (info?.properties?.md_text) {
        this[VALUES] = [info.properties.md_text];
      }
      return configure?.apply(this, arguments);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      const widgets_values = this[VALUES];
      if (widgets_values?.length) {
        var self = this;
        requestAnimationFrame(() => {
          populate.call(self, widgets_values.slice(+(widgets_values.length > 1 && self.inputs?.[0].widget)));
        });
      }
    };
  },
});
