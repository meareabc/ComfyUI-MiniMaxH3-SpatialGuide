import { app } from "../../scripts/app.js";

const DIMENSIONS = ["guide_width", "guide_height", "canvas_width", "canvas_height"];
const HEIGHT = 290;

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function inputValue(node, name, visited = new Set()) {
    if (visited.has(node)) return undefined;
    visited = new Set(visited).add(node);
    const slot = node.inputs?.findIndex((input) => input.name === name) ?? -1;
    if (slot < 0 || node.inputs[slot].link == null) return widget(node, name)?.value;
    const source = node.getInputNode(slot);
    const link = node.getInputLink(slot);
    if (!source || !link) return undefined;
    const type = source.comfyClass || source.type;
    if (type === "ComfySwitchNode") {
        const selected = inputValue(source, "switch", visited);
        return selected == null ? undefined : inputValue(source, selected ? "on_true" : "on_false", visited);
    }
    if (type === "Reroute") return inputValue(source, source.inputs[0].name, visited);
    if (["PrimitiveInt", "PrimitiveBoolean", "PrimitiveNode", "easy int"].includes(type)) {
        return inputValue(source, "value", visited);
    }
    return undefined;
}

function createPositionEditor(node) {
    const root = document.createElement("div");
    root.className = "h3-guide-position";
    root.style.cssText = "width:100%;height:290px;display:flex;flex-direction:column;gap:6px;box-sizing:border-box;padding:6px;color:#dce5ef;font:12px sans-serif;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:218px;flex:1;min-height:100px;border:1px solid #435066;border-radius:5px;box-sizing:border-box;touch-action:none;cursor:grab;";
    canvas.setAttribute("aria-label", "Drag guide position / 拖动参考区域");
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;align-items:center;";
    const status = document.createElement("div");
    status.style.cssText = "white-space:nowrap;font-size:11px;color:#a9b8cb;min-height:14px;";
    const coordinates = document.createElement("span");
    coordinates.style.marginLeft = "auto";
    const context = canvas.getContext("2d");
    let view;
    let drag;
    let liveKey;
    let frame;
    let removed = false;

    function dimensions() {
        const values = DIMENSIONS.map((name, index) => {
            const linked = inputValue(node, name);
            const stored = node.properties?.h3GuideSize?.[index];
            return Number(linked ?? stored ?? widget(node, name)?.value);
        });
        return values.every((value) => Number.isFinite(value) && value > 0) ? values : null;
    }

    function position() {
        return [Number(widget(node, "x").value), Number(widget(node, "y").value)];
    }

    function setPosition(x, y) {
        for (const [name, value] of [["x", x], ["y", y]]) {
            const target = widget(node, name);
            target.value = Math.max(target.options?.min ?? -16384, Math.min(target.options?.max ?? 16384, Math.round(value)));
            target.callback?.(target.value);
        }
        node.graph?.setDirtyCanvas(true, true);
        draw();
    }

    function draw() {
        if (removed) return;
        const bounds = canvas.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const dpr = window.devicePixelRatio || 1;
        const width = Math.round(bounds.width * dpr);
        const height = Math.round(bounds.height * dpr);
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        context.fillStyle = "#111923";
        context.fillRect(0, 0, bounds.width, bounds.height);
        const sizes = dimensions();
        if (!sizes) {
            status.textContent = "请填写参考宽高和画布宽高 / Enter dimensions";
            return;
        }
        const [gw, gh, cw, ch] = sizes;
        const [x, y] = position();
        if (!drag) {
            const left = Math.min(0, x), top = Math.min(0, y);
            const right = Math.max(cw, x + gw), bottom = Math.max(ch, y + gh);
            const scale = Math.min((bounds.width - 36) / (right - left), (bounds.height - 36) / (bottom - top));
            view = { scale, ox: (bounds.width - (right - left) * scale) / 2 - left * scale,
                     oy: (bounds.height - (bottom - top) * scale) / 2 - top * scale };
        }
        const { scale, ox, oy } = view;
        context.fillStyle = "#233247";
        context.fillRect(ox, oy, cw * scale, ch * scale);
        context.strokeStyle = "#7d8fa8";
        context.lineWidth = 1;
        context.strokeRect(ox, oy, cw * scale, ch * scale);
        context.save();
        context.beginPath();
        context.rect(ox, oy, cw * scale, ch * scale);
        context.clip();
        context.strokeStyle = "#384b62";
        for (let i = 1; i < 3; i++) {
            context.beginPath();
            context.moveTo(ox + cw * scale * i / 3, oy);
            context.lineTo(ox + cw * scale * i / 3, oy + ch * scale);
            context.moveTo(ox, oy + ch * scale * i / 3);
            context.lineTo(ox + cw * scale, oy + ch * scale * i / 3);
            context.stroke();
        }
        context.restore();
        context.fillStyle = "rgba(64, 164, 217, 0.35)";
        context.fillRect(ox + x * scale, oy + y * scale, gw * scale, gh * scale);
        context.strokeStyle = "#65d5ff";
        context.lineWidth = 2;
        context.strokeRect(ox + x * scale, oy + y * scale, gw * scale, gh * scale);
        context.fillStyle = "#d2f5ff";
        context.font = "12px sans-serif";
        context.textAlign = "center";
        context.fillText("Guide", ox + (x + gw / 2) * scale, oy + (y + gh / 2) * scale + 4);
        coordinates.textContent = `x ${x} · y ${y}`;
        const unresolved = DIMENSIONS.some((name) => inputValue(node, name) == null);
        status.textContent = `Guide ${gw}×${gh} → ${cw}×${ch}${unresolved ? " · 连线尺寸运行后同步" : ""}`;
    }

    function button(label, action) {
        const element = document.createElement("button");
        element.type = "button";
        element.textContent = label;
        element.style.cssText = "background:#26364b;color:#e1efff;border:1px solid #506783;border-radius:4px;padding:4px 8px;cursor:pointer;";
        element.addEventListener("click", () => {
            const sizes = dimensions();
            if (!sizes) return;
            node.graph?.beforeChange();
            action(sizes);
            node.graph?.afterChange();
        });
        row.appendChild(element);
    }
    button("居中 / Center", ([gw, gh, cw, ch]) => setPosition((cw - gw) / 2, (ch - gh) / 2));
    button("左上 / Origin", () => setPosition(0, 0));
    row.appendChild(coordinates);
    root.append(canvas, row, status);
    root.title = "拖动蓝框调整左上角坐标；按住 Shift 以 32 像素对齐。";
    for (const type of ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick"]) {
        root.addEventListener(type, (event) => event.stopPropagation());
    }
    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || !view || !dimensions()) return;
        const bounds = canvas.getBoundingClientRect();
        const [x, y] = position();
        const [gw, gh] = dimensions();
        const px = (event.clientX - bounds.left - view.ox) / view.scale;
        const py = (event.clientY - bounds.top - view.oy) / view.scale;
        if (px < x || py < y || px > x + gw || py > y + gh) return;
        event.preventDefault();
        node.graph?.beforeChange();
        drag = { id: event.pointerId, x, y, px: event.clientX, py: event.clientY };
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.id) return;
        const step = event.shiftKey ? 32 : 1;
        setPosition(Math.round((drag.x + (event.clientX - drag.px) / view.scale) / step) * step,
                    Math.round((drag.y + (event.clientY - drag.py) / view.scale) / step) * step);
    });
    function endDrag() {
        if (!drag) return;
        drag = null;
        canvas.style.cursor = "grab";
        node.graph?.afterChange();
        draw();
    }
    canvas.addEventListener("lostpointercapture", endDrag);
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    const domWidget = node.addDOMWidget("guide_position_preview", "div", root, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => HEIGHT, getMaxHeight: () => HEIGHT, getHeight: () => HEIGHT,
    });
    domWidget.computeSize = (width) => [width, HEIGHT];
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    for (const name of [...DIMENSIONS, "x", "y"]) {
        const target = widget(node, name);
        const original = target.callback;
        target.callback = function (...args) {
            const result = original?.apply(this, args);
            draw();
            return result;
        };
    }
    // Linked primitives may change without calling this node's widget callbacks.
    function refresh() {
        if (removed) return;
        const bounds = canvas.getBoundingClientRect();
        const key = JSON.stringify([dimensions(), position(), bounds.width, bounds.height]);
        if (key !== liveKey) {
            liveKey = key;
            draw();
        }
        frame = requestAnimationFrame(refresh);
    }
    frame = requestAnimationFrame(refresh);
    node.setSize([Math.max(node.size[0], 390), Math.max(node.size[1], node.computeSize()[1])]);
    return {
        draw,
        remove() {
            removed = true;
            cancelAnimationFrame(frame);
            observer.disconnect();
        },
    };
}

app.registerExtension({
    name: "MiniMaxH3.SpatialGuidePosition",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3GuidePosition") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.h3PositionEditor = createPositionEditor(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            requestAnimationFrame(() => this.h3PositionEditor?.draw());
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            const result = executed?.call(this, message, ...args);
            if (message.guide_size && message.canvas_size) {
                this.properties ??= {};
                this.properties.h3GuideSize = [...message.guide_size, ...message.canvas_size];
                this.h3PositionEditor?.draw();
            }
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this.h3PositionEditor?.remove();
            return removed?.apply(this, args);
        };
    },
});
