# Dataset: FracAtlas

## Source
- Title: FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs
- DOI: `10.6084/m9.figshare.22363012.v6`
- Figshare API: `https://api.figshare.com/v2/articles/22363012`
- Archive file: `FracAtlas.zip`
- License: `CC BY 4.0`

## Local Layout

Raw download/extraction:
- `data/raw/FracAtlas.zip`
- `data/raw/fracatlas/FracAtlas/`

Prepared outputs:
- `data/processed/fracatlas/masks/*.png`
- `data/processed/fracatlas/manifests/all.csv`
- `data/processed/fracatlas/manifests/train.csv`
- `data/processed/fracatlas/manifests/val.csv`
- `data/processed/fracatlas/manifests/test.csv`

## Commands

```powershell
python .\scripts\download_fracatlas.py
python .\scripts\prepare_fracatlas_segmentation.py
```

## Notes
- The current preparation script rasterizes polygon regions from `VGG_fracture_masks.json`.
- For non-fractured cases, the mask is all zeros.
- Paths in manifests are repo-relative so the project remains portable across machines.
