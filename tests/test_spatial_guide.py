import asyncio
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import comfy.cli_args
comfy.cli_args.args.cpu = True

import comfy.conds
import execution
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, patchify_video
from comfy.model_patcher import ModelPatcher
from comfy.nested_tensor import NestedTensor


spec = importlib.util.spec_from_file_location("h3_spatial_guide", Path(__file__).resolve().parents[1] / "__init__.py")
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)


def segment_positions(layout, kind):
    return [layout.position_ids[a:b] for a, b, k in layout.segments if k == kind]


class SpatialGuideTests(unittest.TestCase):
    def test_native_size_and_offsets_match_target_region(self):
        for x in (0, 160, 320):
            keyframes = [{"resolved_frame_index": 0, "latent": torch.empty(1, 24, 7, 30, 40), guide.GUIDE_POSITION: (x, 0)}]
            native = PackedLayout(8, 12, 30, 60, 65, keyframes=keyframes)
            result = guide.spatial_guide_layout(native, keyframes)
            target = segment_positions(native, "video")[0].reshape(12, 15, 30, 3)
            expected = target[:7, :, x // 32:x // 32 + 20].reshape(-1, 3)
            torch.testing.assert_close(segment_positions(result, "cond")[0], expected, rtol=0, atol=1e-13)
            self.assertEqual(segment_positions(result, "cond")[0].shape[0], patchify_video(keyframes[0]["latent"]).shape[0])
            for kind in ("text", "audio", "video"):
                torch.testing.assert_close(segment_positions(result, kind)[0], segment_positions(native, kind)[0], rtol=0, atol=0)

    def test_same_size_is_native_equivalent(self):
        keyframes = [{"resolved_frame_index": 3, "latent": torch.empty(1, 24, 2, 6, 10), guide.GUIDE_POSITION: (0, 0)}]
        native = PackedLayout(4, 7, 6, 10, 37, keyframes=keyframes)
        result = guide.spatial_guide_layout(native, keyframes)
        self.assertEqual(result.segments, native.segments)
        self.assertEqual(result.signature, native.signature)
        for name in ("position_ids", "img_pos", "img_update", "audio_pos", "audio_update"):
            torch.testing.assert_close(getattr(result, name), getattr(native, name), rtol=0, atol=1e-13)

    def test_mixed_guides_audio_references_and_fractional_offsets(self):
        keyframes = [
            {"resolved_frame_index": 0, "latent": torch.empty(1, 24, 2, 4, 6), guide.GUIDE_POSITION: (17, 3), "audio_latent": torch.empty(1, 32, 2, 8)},
            {"resolved_frame_index": 5, "latent": torch.empty(1, 24, 1, 6, 10)},
            {"resolved_frame_index": 7, "latent": torch.empty(1, 24, 1, 8, 12), guide.GUIDE_POSITION: (-4, 9)},
        ]
        refs = [{"kind": "image", "latent_h": 4, "latent_w": 4}]
        native = PackedLayout(4, 7, 6, 10, 37, keyframes=keyframes, refs=refs)
        before = native.position_ids.clone()
        result = guide.spatial_guide_layout(native, keyframes)
        for original, changed in zip(native.segments, result.segments):
            if original[2] != "cond":
                torch.testing.assert_close(native.position_ids[original[0]:original[1]], result.position_ids[changed[0]:changed[1]], rtol=0, atol=0)
        original_guides = segment_positions(native, "cond")
        new_guides = segment_positions(result, "cond")
        torch.testing.assert_close(new_guides[1], original_guides[1], rtol=0, atol=0)
        expected_offset = torch.tensor([3, 17], dtype=torch.float64) * 2 / math.sqrt(6 * 10)
        torch.testing.assert_close(new_guides[0][0, 1:] - original_guides[0][0, 1:], expected_offset)
        self.assertEqual([p.shape[0] for p in new_guides], [12, 15, 24])
        self.assertEqual(result.img_update.sum().item(), 7 * 15)
        self.assertEqual(result.img_pos.numel() + result.audio_pos.numel() + 4, result.seq_len)
        torch.testing.assert_close(native.position_ids, before, rtol=0, atol=0)

    def test_node_preserves_pixels_and_model_patches_are_scoped(self):
        model = ModelPatcher(torch.nn.Module(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        images = torch.rand(24, 48, 65, 3)
        encoded = []

        def encode(frames):
            encoded.append(frames)
            return torch.zeros(1, 24, 7, frames.shape[1] // 16, frames.shape[2] // 16)

        positive = [[torch.zeros(1, 2, 8), {}], [torch.zeros(1, 3, 8), {"minimax_keyframes": [{"resolved_frame_index": 0}]}]]
        latent = {"samples": NestedTensor([torch.zeros(1, 24, 12, 6, 10), torch.zeros(1, 32, 2, 65)])}
        patched, output = guide.MiniMaxH3AddSpatialGuide.execute(model, positive, SimpleNamespace(encode=encode), latent, images, frame_idx=-39, x=17, y=3)
        self.assertEqual(encoded[0].shape, (22, 64, 96, 3))
        torch.testing.assert_close(encoded[0][:, :48, :65], images[:22], rtol=0, atol=0)
        torch.testing.assert_close(encoded[0][:, :48, 65:], images[:22, :, -1:].expand(-1, -1, 31, -1), rtol=0, atol=0)
        self.assertNotIn("minimax_keyframes", positive[0][1])
        self.assertEqual(len(positive[1][1]["minimax_keyframes"]), 1)
        self.assertEqual([len(c[1]["minimax_keyframes"]) for c in output], [1, 2])
        self.assertNotIn("noise_mask", latent)
        self.assertEqual(model.wrappers, {})
        chained = guide.patch_spatial_guides(patched)
        wrapper_type = comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
        self.assertEqual(len(chained.get_wrappers(wrapper_type, guide.GUIDE_POSITION)), 1)
        keyframes = output[0][1]["minimax_keyframes"]
        native = PackedLayout(2, 12, 6, 10, 65, keyframes=keyframes)
        payload = {"keyframes": keyframes, "layout": native}
        negative = {"layout": native}
        sampler_model = SimpleNamespace(conds={
            "positive": [{"model_conds": {"minimax_payload": comfy.conds.CONDConstant(payload)}}],
            "negative": [{"model_conds": {"minimax_payload": comfy.conds.CONDConstant(negative)}}],
        })
        executor = comfy.patcher_extension.WrapperExecutor.new_executor(
            lambda wrapped, value: value, chained.get_wrappers(wrapper_type, guide.GUIDE_POSITION))
        self.assertEqual(executor.execute(sampler_model, 123), 123)
        result = sampler_model.conds["positive"][0]["model_conds"]["minimax_payload"].cond["layout"]
        self.assertEqual(segment_positions(result, "cond")[0].shape[0], 7 * 2 * 3)
        self.assertIs(payload["layout"], native)
        self.assertIs(sampler_model.conds["negative"][0]["model_conds"]["minimax_payload"].cond, negative)

    def test_small_native_model_accepts_local_guide(self):
        torch.manual_seed(3)
        model = MiniMaxH3Model(hidden_size=8, num_layers=1, token_refiner_num_layers=0,
                              num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=16,
                              timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
                              text_dim=8, rope_inv_freq_len=1, dtype=torch.float32,
                              device=torch.device("cpu"), operations=torch.nn)
        model.rope.inv_freq.fill_(1.0)
        video = torch.randn(1, 24, 2, 4, 6)
        audio = torch.randn(1, 32, 2, 5)
        context = torch.randn(1, 3, 8)
        keyframes = [{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 2, 2, 4), guide.GUIDE_POSITION: (32, 16)}]
        native = PackedLayout(3, 2, 4, 6, 5, keyframes=keyframes)
        payload = {"layout": guide.spatial_guide_layout(native, keyframes), "cond_video_latents": [keyframes[0]["latent"]]}
        with patch("comfy.model_management.in_training", True):
            result = model._forward([video, audio], torch.tensor([500.0]), context, minimax_payload=payload)
        for actual, original in zip(result, (video, audio)):
            self.assertEqual(actual.shape, original.shape)
            self.assertTrue(torch.isfinite(actual).all())

    def test_extension_registers_v3_node(self):
        extension = asyncio.run(guide.comfy_entrypoint())
        self.assertEqual(asyncio.run(extension.get_node_list()), [guide.MiniMaxH3AddSpatialGuide, guide.MiniMaxH3GuidePosition])
        self.assertEqual(guide.MiniMaxH3AddSpatialGuide.define_schema().node_id, "MiniMaxH3AddSpatialGuide")

    def test_position_outputs_pixel_coordinates_unchanged(self):
        result = guide.MiniMaxH3GuidePosition.execute(640, 480, 960, 480, 160, -17)
        self.assertEqual(result.result, (160, -17))
        self.assertEqual(result.ui, {"guide_size": [640, 480], "canvas_size": [960, 480]})
        schema = guide.MiniMaxH3GuidePosition.define_schema()
        self.assertEqual([output.display_name for output in schema.outputs], ["x", "y"])
        self.assertTrue(schema.is_output_node)

    def test_position_previews_first_frame_and_actual_canvas_size(self):
        images = torch.zeros(2, 48, 65, 3)
        images[0, ..., 0] = 1
        images[1, ..., 1] = 1
        latent = {"samples": NestedTensor([torch.zeros(1, 24, 2, 35, 16), torch.zeros(1, 32, 2, 8)])}
        with tempfile.TemporaryDirectory() as directory, patch("folder_paths.get_temp_directory", return_value=directory):
            result = guide.MiniMaxH3GuidePosition.execute(320, 416, 270, 560, -38, 147, image=images, latent=latent)
            self.assertEqual(result.result, (-38, 147))
            self.assertEqual(result.ui["guide_size"], [65, 48])
            self.assertEqual(result.ui["canvas_size"], [256, 560])
            self.assertEqual(len(result.ui["guide_preview"]), 1)
            descriptor = result.ui["guide_preview"][0]
            self.assertEqual(descriptor["type"], "temp")
            with Image.open(Path(directory) / descriptor["subfolder"] / descriptor["filename"]) as image:
                self.assertEqual(image.size, (65, 48))
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_position_accepts_single_image_preview(self):
        images = torch.ones(1, 32, 64, 3)
        with tempfile.TemporaryDirectory() as directory, patch("folder_paths.get_temp_directory", return_value=directory):
            result = guide.MiniMaxH3GuidePosition.execute(1, 1, 128, 96, 32, 32, image=images)
            self.assertEqual(result.ui["guide_size"], [64, 32])
            self.assertEqual(result.ui["canvas_size"], [128, 96])

    def test_partial_execution_refreshes_preview_without_downstream_output(self):
        events = []
        server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda event, data, client=None: events.append((event, data)))
        executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
        prompt = {
            "source": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 48, "batch_size": 2, "color": 0xFF0000}},
            "position": {"class_type": "MiniMaxH3GuidePosition", "inputs": {
                "guide_width": 320, "guide_height": 416, "canvas_width": 128, "canvas_height": 96,
                "x": 32, "y": 16, "image": ["source", 0],
            }},
            "downstream": {"class_type": "PreviewImage", "inputs": {"images": ["position", 0]}},
        }
        with tempfile.TemporaryDirectory() as directory, patch("folder_paths.get_temp_directory", return_value=directory), patch.dict(
            guide.nodes.NODE_CLASS_MAPPINGS, {"MiniMaxH3GuidePosition": guide.MiniMaxH3GuidePosition}
        ):
            valid = asyncio.run(execution.validate_prompt("position-preview", prompt, ["position"]))
            self.assertTrue(valid[0], valid)
            self.assertEqual(set(valid[2]), {"position"})
            for prompt_id in ("position-preview", "position-preview-cached"):
                events.clear()
                executor.execute(prompt, prompt_id, {"client_id": "position-test"}, valid[2])
                self.assertTrue(executor.success, executor.status_messages)
                output = executor.history_result["outputs"]["position"]
                self.assertEqual(output["guide_size"], [64, 48])
                self.assertEqual(len(output["guide_preview"]), 1)
                self.assertNotIn("downstream", executor.history_result["outputs"])
                self.assertTrue(any(event == "executed" and data["node"] == "position" for event, data in events))


if __name__ == "__main__":
    unittest.main()
