# DINOv3 Canopy Height Model — Fine-tuned Inference

Runs a DINOv3 ViT-L/16 backbone (`facebook/dinov3-vitl16-pretrain-sat493m`) with a
custom-trained convolutional depth head to predict canopy height maps (CHM) from
Planet 8-band SuperDove surface-reflectance imagery. The head was fine-tuned on
Maxar imagery + boreal Alaska lidar CHM labels, then applied to Planet imagery
using NIR/Red/Green input bands (see `test_inference.ipynb`).

## 1. Set up the environment

```
conda env create -f environment.yaml
conda activate dino-chm-inference
```

See the comments at the top of `environment.yaml` for why torch/torchvision are
installed via pip rather than conda.

## 2. Get Hugging Face access

The backbone (`facebook/dinov3-vitl16-pretrain-sat493m`) is a gated model:

1. Create a free account at https://huggingface.co if you don't have one.
2. While logged in, request/accept access at
   https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m.
3. Authenticate this environment:
   ```
   python -c "from huggingface_hub import login; login()"
   ```
   Paste your token when prompted (input is hidden). This writes to
   `~/.cache/huggingface/token` and only needs to be done once.

## 3. Get the checkpoint

`best_dinov3_fulldataset_customloss.pth` (~1.5GB) is **not** included in this repo
(too large for git). It was sent separately — place it at:

```
./checkpoint/best_dinov3_fulldataset_customloss.pth
```

## 4. Prepare your imagery

The pipeline expects 8-band Planet SuperDove (PSB.SD) surface-reflectance scenes,
with the standard band order/numbering (1-indexed, rasterio convention):

| Band | Name         |
|------|--------------|
| 1    | Coastal Blue |
| 2    | Blue         |
| 3    | Green I      |
| 4    | Green        |
| 5    | Yellow       |
| 6    | Red          |
| 7    | Red Edge     |
| 8    | NIR          |

Only bands 4 (Green), 6 (Red), and 8 (NIR) are actually used (as NRG input, matching
how the model was fine-tuned). NoData is assumed to be `0`, matching Planet SR's fill
convention. If your imagery uses a different sensor, band count/order, or NoData
convention, `np_stats.ipynb` and `test_inference.ipynb` will need those constants
updated (`PLANET_BAND_NAMES`, `NRG_BAND_NUMBERS`/band indices, `nodata=0`).

Point your scenes at `./data/planet_scenes/` (or edit `INPUT_DIR` in both notebooks
to wherever they actually live).

## 5. Compute normalization stats for your own imagery

Run **`np_stats.ipynb`** first. It samples pixels from your `INPUT_DIR` scenes and
writes normalization statistics to `./numpy_stats/`:

- Per-band min/max/mean/std/mean+3σ over the raw 8 bands (used to set NIR/Red/Green
  minmax-scaling bounds)
- Mean/std of the NIR/Red/Green bands *after* that [0,1] scaling (used for the
  z-score normalization step)

**Do not reuse stats computed on a different dataset** — they were fit to the
specific reflectance distribution of the imagery they were sampled from, and using
stats from a different region/time period/sensor will silently produce wrong
predictions rather than an obvious error.

## 6. Run inference

Run **`test_inference.ipynb`**. It tiles each scene, runs it through the model, and
writes predictions to `./output/` as single-band int16 GeoTIFFs:

- Units: **decimeters** (height_m × 10, rounded)
- NoData: `-9999`
- Filename: derived from the input scene name, suffixed `sr-02m.chm.tif`

## Notes / limitations

- This checkpoint was fine-tuned on boreal Alaska burn-area imagery. Predictions on
  substantially different biomes, canopy structures, or imagery conditions haven't
  been validated and should be treated with appropriate skepticism.
- The backbone is frozen; only the depth head was trained. Output resolution is
  limited by the 64×64 input chip size used during fine-tuning (a 4×4 grid of
  16×16-pixel patch tokens), which tends to produce smoother/blurrier maps than
  higher-resolution dense-prediction architectures.
