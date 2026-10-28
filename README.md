# MiniMax H3 Spatial Guide

Native-size image/video guides and a drag-to-position editor for ComfyUI's native MiniMax H3 model. No additional Python dependencies.

![Drag the guide rectangle to choose its position](docs/position-editor.png)

## Install

Place this repository in `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-SpatialGuide`, restart ComfyUI, and refresh the browser. Requires a ComfyUI version with native MiniMax H3 support. If installing by URL through Manager, use this repository's Git URL.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/meareabc/ComfyUI-MiniMaxH3-SpatialGuide.git
```

## Nodes

**Add Spatial Guide for MiniMax H3**

- Connect H3 `model`, `positive`, video `vae`, target AV `latent`, and guide frames `image`.
- The guide keeps its input dimensions. `x` and `y` locate its **top-left corner** in target pixels: positive x moves right, positive y moves down.
- Connect **both output MODEL and positive** to the sampler/guider. Connect the target latent directly to the sampler.
- For audio, chain a native Add Guide after this node with only its audio inputs connected.

**MiniMax H3 Guide Position**

- Connect the same `image` batch used by Spatial Guide to preview its **first frame**. This also works with a single image. The actual image dimensions override the guide size fields for preview.
- Connect the target AV `latent` to use its actual canvas dimensions. This accounts for alignment performed by the empty latent node.
- Click **Preview** to run this node and its upstream dependencies without running downstream sampling. Then drag the image; coordinates and placement update immediately. Hold Shift to snap movement to 32 pixels. Center and Origin buttons provide quick alignment.
- Connect its `x` and `y` outputs to Add Spatial Guide's corresponding inputs. Coordinates save with the workflow and can also be edited numerically.
- After changing the image source or target size, refresh the preview. Old dimensions are invalidated instead of being combined with new input values. If the guide is actually larger than the canvas, the preview shows the overflow without resizing it.
- Without `image` or `latent`, set `guide_width`, `guide_height`, `canvas_width`, and `canvas_height` manually, or connect their INT inputs.

For a 640×480 guide on a 960×480 canvas, `x=0,y=0` leaves 320 pixels to the right; `x=160,y=0` centers it. For a 480×640 guide on a 480×960 canvas, use `x=0,y=160` to center it vertically.

## Behavior and validation

Video guides follow native Add Guide's valid lengths: 5, 22, 39… frames; batches below 5 frames use the first image. Nonmultiples of 32 are padded at the right/bottom with replicated edges, without scaling. This is positional conditioning, so generated pixels can change inside the guide region.

CPU tests cover native-size coordinates, mixed guides/audio/references, clone isolation, registration, first-frame selection, actual latent dimensions, partial execution and cached previews, and a small native H3 forward. Browser checks cover image display, dragging, zoom, and stale input invalidation. Full pretrained outpainting quality has not been validated.

```bash
python -m unittest discover -s custom_nodes/ComfyUI-MiniMaxH3-SpatialGuide/tests -v
```

## 中文使用说明

重启并刷新后搜索 **MiniMax H3 Guide Position**，把同一份 Guide 视频／图片接入 `image`，目标 AV latent 接入 `latent`。点击“刷新预览”即可单独执行预览及其上游准备，显示视频第一帧或所选图片和真实画布尺寸；随后拖动图片，实时调整位置。修改来源或画布尺寸后再次刷新。

输出的 `x/y` 接到 **Add Spatial Guide for MiniMax H3**，表示参考区域左上角坐标，原点在目标画布左上角。按住 Shift 拖动可按 32 像素对齐，也可点击居中或左上归零。如果参考区域确实大于目标画布，预览会如实显示越界，需要增大画布宽高。

Spatial Guide 的 **MODEL 和 positive 两个输出都要连接到采样器**，目标 latent 直接接采样器。音频可用原生 Add Guide 单独添加。
