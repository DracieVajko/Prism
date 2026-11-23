# Final results — 2026-11-23 (after the colour-cast and restore fixes)

Machine: RTX 3060 Laptop 6 GB, 32 GB RAM, Windows. LM Studio unloaded.

## Colorization — colour cast removed

Both DDColor checkpoints were adding a constant chroma offset to the whole
image, which is exactly what "red-ish" and "yellow-ish" look like:

| Checkpoint | Measured bias before | After fix |
|---|---|---|
| ddcolor-full | a ≈ +7.5 (red) | a ≈ -0.2, b ≈ -0.1 |
| ddcolor-artistic | b ≈ +12.9 (yellow) | a ≈ 0.0, b ≈ -0.5 |

The fix subtracts the per-image median of each a/b channel before scaling, so
the scene is neutral while local colour is untouched. The old overall
percentile stretch (which oversaturated) and the extra HSV boost are gone.

**Red/blue "3D glasses" fringing** came from per-pixel chroma noise in the
ddcolor-full checkpoint being amplified and then upscaled with plain bilinear.
A guided filter keyed on the model-resolution luminance now cleans the chroma
before the upscale. Measured as the mean |Laplacian| of the a/b channels:

| Model | Fringe before | Fringe after |
|---|---|---|
| ddcolor-full | a 2.28, b 2.74 | **a 0.94, b 0.81** |
| ddcolor-artistic | a 1.23, b 1.28 | a 0.75, b 0.57 |
| deoldify-artistic | a 1.57, b 1.20 | unchanged (already good) |

Chroma after the filter is 13.0 (full) / 18.5 (artistic) / 36.3 (deoldify), so
the colours stay as strong as before while the fringing is gone.

Input for all three: [old_bw_photo_ffmpeg.jpg](file:///D:/AI/Prism/testdata/images/old_bw_photo_ffmpeg.jpg)

| Model | Time | Chroma | Residual cast (a/b) | Output |
|---|---|---|---|---|
| ddcolor-full | 4.6 s | 14.0 | -0.2 / -0.1 | [final_colorize_ddcolor-full.png](file:///D:/AI/Prism/testdata/images/final_colorize_ddcolor-full.png) |
| ddcolor-artistic | 3.3 s | 18.8 | 0.0 / -0.5 | [final_colorize_ddcolor-artistic.png](file:///D:/AI/Prism/testdata/images/final_colorize_ddcolor-artistic.png) |
| deoldify-artistic | 5.6 s | 36.3 | -5.8 / -10.7 | [final_colorize_deoldify-artistic.png](file:///D:/AI/Prism/testdata/images/final_colorize_deoldify-artistic.png) |

deoldify keeps the most chroma and was left untouched — it was the best result
by review. Its small cool offset comes from the model itself, not from a bug.

## Logo generation — best settings

`amazing-logos-v3` already ran at its card values (30 steps, CFG 7.5). The
"slopped" look came from the prompt and a missing negative prompt. Now:

- prompt: subject + `vector art, 2d, centered, white background, best quality, sharp clean edges`
- negative: `low quality, worst quality, bad composition, text, watermark, asymmetric, photo, realistic, 3d render`

| Size | Time | Output |
|---|---|---|
| 512x512 | 6.6 s | [logo_best_512-512.png](file:///D:/AI/Prism/testdata/images/logo_best_512-512.png) |
| 768x768 | 17.3 s | [logo_best_768-768.png](file:///D:/AI/Prism/testdata/images/logo_best_768-768.png) |

## Restoration — Z-Image first, SDXL only as fallback

`POST /v1/images/restore` now prefers Z-Image Turbo (keeps the composition) and
falls back to Juggernaut img2img with the strength clamped to ≤ 0.35, because
0.5 redrew the scene into a different photograph.

| Engine | Time | Output |
|---|---|---|
| z-image-turbo (default when installed) | 435 s | [restore_best_zimage.png](file:///D:/AI/Prism/testdata/images/restore_best_zimage.png) |
| Juggernaut XL img2img (fallback) | — | [final_restore_juggernaut.png](file:///D:/AI/Prism/testdata/images/final_restore_juggernaut.png) |

Override with `PRISM_RESTORE_ENGINE=sd` to force the SDXL path,
`PRISM_RESTORE_MODEL=<id>` to change the checkpoint.

## Image generation

| Model | Time | Output |
|---|---|---|
| NoobAI-XL-Vpred 1024x1024 | 37.8 s | [final_NoobAI-XL-Vpred.png](file:///D:/AI/Prism/testdata/images/final_NoobAI-XL-Vpred.png) |
| amazing-logos-v3 | see above | see above |
| Juggernaut XL v9 | 136 s | [out_JuggernautXL_v9.png](file:///D:/AI/Prism/testdata/images/out_JuggernautXL_v9.png) |
| DreamShaper 8 | 5.7 s | [out_dreamshaper_8.png](file:///D:/AI/Prism/testdata/images/out_dreamshaper_8.png) |
