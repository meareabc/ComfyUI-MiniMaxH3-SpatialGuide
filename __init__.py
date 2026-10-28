import copy
import math

import torch
import torch.nn.functional as F

import comfy.conds
import comfy.patcher_extension
import nodes
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_api.latest import ComfyExtension, io, ui


GUIDE_POSITION = "minimax_h3_spatial_guide"
WEB_DIRECTORY = "./web"


def spatial_guide_layout(native, keyframes):
    _, _, height, width, _ = native.signature
    frame_rows = (height // 2) * (width // 2)
    spacing = 64.0 / math.sqrt(height * width)
    guides = iter(kf for kf in keyframes if kf.get("latent") is not None)
    position_parts, segments = [], []
    img_pos, img_update, audio_pos, audio_update = [], [], [], []
    cursor = 0
    for start, stop, kind in native.segments:
        positions = native.position_ids[start:stop]
        if kind == "cond":
            guide = next(guides)
            if GUIDE_POSITION in guide:
                x, y = guide[GUIDE_POSITION]
                _, _, time, h, w = guide["latent"].shape
                grid = positions.new_empty(time, h // 2, w // 2, 3)
                grid[..., 0] = positions[::frame_rows, 0][:, None, None]
                hh = torch.arange(h // 2, dtype=positions.dtype, device=positions.device)
                ww = torch.arange(w // 2, dtype=positions.dtype, device=positions.device)
                grid[..., 1] = positions[0, 1] + (hh[:, None] + y / 32.0) * spacing
                grid[..., 2] = positions[0, 2] + (ww[None, :] + x / 32.0) * spacing
                positions = grid.reshape(-1, 3)
        count = positions.shape[0]
        position_parts.append(positions)
        segments.append((cursor, cursor + count, kind))
        if kind in ("cond", "ref_img", "video"):
            img_pos.append(torch.arange(cursor, cursor + count))
            img_update.append(torch.full((count,), kind == "video", dtype=torch.bool))
        elif kind in ("cond_audio", "ref_audio", "audio"):
            audio_pos.append(torch.arange(cursor, cursor + count))
            audio_update.append(torch.full((count,), kind == "audio", dtype=torch.bool))
        cursor += count

    layout = copy.copy(native)
    layout.position_ids = torch.cat(position_parts)
    layout.segments = segments
    layout.seq_len = cursor
    layout.img_pos = torch.cat(img_pos)
    layout.img_update = torch.cat(img_update)
    layout.audio_pos = torch.cat(audio_pos)
    layout.audio_update = torch.cat(audio_update)
    return layout


def spatial_guide_sample(executor, model_wrap, *args, **kwargs):
    for conditions in model_wrap.conds.values():
        for condition in conditions:
            model_conds = condition["model_conds"]
            payload = model_conds["minimax_payload"].cond
            keyframes = payload.get("keyframes", [])
            if any(GUIDE_POSITION in kf for kf in keyframes):
                payload = dict(payload, layout=spatial_guide_layout(payload["layout"], keyframes))
                model_conds["minimax_payload"] = comfy.conds.CONDConstant(payload)
    return executor(model_wrap, *args, **kwargs)


def patch_spatial_guides(model):
    patched = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
    if not patched.get_wrappers(wrapper_type, GUIDE_POSITION):
        patched.add_wrapper_with_key(wrapper_type, GUIDE_POSITION, spatial_guide_sample)
    return patched


class MiniMaxH3AddSpatialGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AddSpatialGuide",
            display_name="Add Spatial Guide for MiniMax H3",
            category="model/conditioning/minimax",
            description="Place an image or video guide at its original size within the target canvas. Connect both MODEL and positive outputs to the sampler.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("image", tooltip="Guide frames at their original size. Video uses 5, 22, 39... frames, matching the native Add Guide node."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999),
                io.Int.Input("x", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION,
                             tooltip="Guide's left edge in target pixels. Use half the width difference to center it."),
                io.Int.Input("y", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION,
                             tooltip="Guide's top edge in target pixels. Use half the height difference to center it."),
            ],
            outputs=[io.Model.Output(), io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, model, positive, vae, latent, image, frame_idx=0, x=0, y=0) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3AddSpatialGuide expects a MiniMax H3 AV latent")
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(samples.tensors[0].shape[2]))
        guide_frames = image.shape[0]
        guide_frames = 1 if guide_frames < 5 else guide_frames - (guide_frames - 5) % 17
        resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved < 0 or resolved + guide_frames > frame_count:
            raise ValueError("a {} frame guide at frame_idx {} does not fit in the video's {} frames".format(guide_frames, frame_idx, frame_count))

        frames = image[:guide_frames, ..., :3]
        height, width = frames.shape[1:3]
        # Complete VAE/DiT patches without scaling or cropping the source.
        if height % 32 or width % 32:
            frames = F.pad(frames.movedim(-1, 1), (0, -width % 32, 0, -height % 32), mode="replicate").movedim(1, -1)
        keyframe = {"resolved_frame_index": resolved, "latent": vae.encode(frames), GUIDE_POSITION: (x, y)}
        positive = [[cond, {**options, "minimax_keyframes": list(options.get("minimax_keyframes", [])) + [keyframe]}]
                    for cond, options in positive]
        return io.NodeOutput(patch_spatial_guides(model), positive)


class MiniMaxH3GuidePosition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3GuidePosition",
            display_name="MiniMax H3 Guide Position",
            category="model/conditioning/minimax",
            description="Preview the guide's first frame and drag it within the target canvas. Run this node to refresh input dimensions without sampling.",
            is_output_node=True,
            inputs=[
                io.Int.Input("guide_width", default=640, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("guide_height", default=480, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("canvas_width", default=960, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("canvas_height", default=480, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("x", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, socketless=True),
                io.Int.Input("y", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, socketless=True),
                io.Image.Input("image", optional=True, tooltip="Connect the same image/video batch as Spatial Guide. The first frame determines the guide preview and its actual size."),
                io.Latent.Input("latent", optional=True, tooltip="Target H3 AV latent. Uses its actual canvas dimensions, including the latent's size alignment."),
            ],
            outputs=[io.Int.Output(display_name="x"), io.Int.Output(display_name="y")],
        )

    @classmethod
    def execute(cls, guide_width, guide_height, canvas_width, canvas_height, x, y, image=None, latent=None) -> io.NodeOutput:
        preview = {}
        if image is not None:
            guide_height, guide_width = image.shape[1:3]
            preview["guide_preview"] = ui.PreviewImage(image[:1, ..., :3]).as_dict()["images"]
        if latent is not None:
            samples = latent["samples"]
            if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
                raise ValueError("MiniMaxH3GuidePosition expects a MiniMax H3 AV latent")
            canvas_height, canvas_width = [size * 16 for size in samples.tensors[0].shape[-2:]]
        preview.update({"guide_size": [guide_width, guide_height], "canvas_size": [canvas_width, canvas_height]})
        return io.NodeOutput(x, y, ui=preview)


class MiniMaxH3SpatialGuideExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AddSpatialGuide, MiniMaxH3GuidePosition]


async def comfy_entrypoint():
    return MiniMaxH3SpatialGuideExtension()
