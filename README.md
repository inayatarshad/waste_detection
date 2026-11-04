# waste_detection

Detecting waste dumps, and plastic waste in particular, in high-resolution overhead imagery of
**Almeria, Spain**.

## Why Almeria is hard

Almeria has the largest concentration of plastic greenhouses in the world. From above, large parts of
the province look like plastic. A model that learns "plastic means waste" flags every greenhouse.
The models here must therefore learn the difference between greenhouse plastic, which is not waste, and
plastic that has been dumped in a field, a rambla or next to a road, which is waste. Almeria-specific
labels and a greenhouse false-alarm metric (notebook 04) exist for exactly this reason.

## Imagery

The models are built for very-high-resolution RGB images at **20-60 cm per pixel**. That includes aerial
orthophotos and very-high-resolution satellites such as WorldView, Pleiades Neo or Google Earth exports.
At 10 m per pixel (Sentinel-2) a whole dump covers only a few pixels, so it cannot be used for this task.

For scanning Almeria we use the free PNOA orthophotos from the Spanish IGN, which are 25-50 cm, served
over WMS and licensed CC BY 4.0. Any other image of similar resolution can be scanned the same way
with `scan.scan_image`.

## Approach

| Step | What | Data | Model |
|---|---|---|---|
| 1 | Waste / no waste for each 210 m x 210 m window, plus waste type (**Plastic**, rubble, bulky items, scrap, tires, firewood, vehicles) | AerialWaste v3: 11,703 images labelled by professional photo interpreters (ARPA Lombardia) | ConvNeXt-Tiny, multi-task heads, unknown type labels masked |
| 2 | Boxes around dump sites | Roboflow "Aerial-Dumping-Sites" v6 | Faster R-CNN ResNet-50-FPN v2 |
| 3 | Scan an Almeria area and list the candidate sites | PNOA orthophotos | models 1 + 2 |
| 4 | Label a sample of Almeria windows, then fine-tune and measure on separate 2 km blocks | own labels | model 1, fine-tuned |

### Evaluation rules

These are the same rules as the GSTP hazard models.

- The best epoch and every decision threshold are chosen on **validation** data. The **test** split is scored once, and the code refuses to score it a second time.
- Leakage is checked:
  - Roboflow's augmented copies of the same photo are kept together when splitting.
  - Training images that are near-duplicates of test images are removed.
  - In Almeria, train, validation and test are separate 2 km blocks.
- Every model is compared with a **classic method** scored on the same images:
  - hand-crafted colour, edge and texture features with logistic regression;
  - a "white plastic" pixel rule for the plastic type.
- The detector's test set is the same 63 images used for the earlier YOLOv8s baseline (mAP50 0.42), so the two can be compared directly.
- Almeria numbers come with 95 % bootstrap intervals, because local test sets are small.

## Run order (Google Colab, T4 GPU)

| Notebook | Time | What you need |
|---|---|---|
| `01_aerialwaste_classifier` | ~30 min download (first time only) + ~1.5 h training | nothing (Zenodo is public) |
| `02_dumpsite_detector` | ~1 h | a Roboflow API key |
| `03_almeria_scan_and_label` | minutes for the demo area | a few hours of labelling (in-notebook buttons) |
| `04_almeria_finetune` | ~30 min | the labelled sheet from 03 |

To open a notebook:
`colab.research.google.com/github/inayatarshad/waste_detection/blob/main/notebooks/<name>.ipynb`

Choose **Runtime > Change runtime type > T4 GPU**, run all cells, and click "Connect to Google Drive" when
asked. All outputs (checkpoints, ONNX files, reports, figures, scan maps) go to
`MyDrive/waste_runs/`.

## Repository

```
configs/      Almeria scan areas
notebooks/    01-04 (Colab)
src/wastedet/ aerialwaste.py   dataset reader, splits, augmentation
              dumpsites.py     YOLO-box reader, grouped splits, box augmentation
              models.py        classifier (+ attention map), detector, ONNX export
              metrics.py       AP, thresholds, detection mAP, bootstrap intervals
              baselines.py     classic methods
              train_cls.py     classifier training / one-time test
              train_det.py     detector training / one-time test
              geo.py           coordinates, PNOA WMS, scan grid, block splits
              scan.py          window scanning, candidate sites, GeoJSON, labelling sheet
              almeria.py       Almeria labels and dataset mixing
              plots.py         figures
tests/        pytest suite on synthetic data (no downloads needed): python -m pytest -q
```

## Data and licences

- **AerialWaste**: Torres & Fraternali, *Scientific Data* 10, 63 (2023), Zenodo record 12607190. The Zenodo record is CC BY 4.0, but some images come from Google Earth, and their use must follow Google's terms. The authors' own toolkit and model repositories are CC BY-NC-ND. We do not use their code or weights; everything here is written from scratch.
- **Aerial-Dumping-Sites v6**: Roboflow Universe, CC BY 4.0, with attribution to the dataset authors.
- **PNOA**: "PNOA cedido por (c) Instituto Geografico Nacional de Espana", CC BY 4.0.
- **Libraries**: PyTorch / torchvision (BSD) and timm (Apache-2.0). Ultralytics YOLO is deliberately not used, because it is AGPL-3.0.

**Before any commercial use**, check the Google Earth imagery terms that apply to the AerialWaste subset.

## Limitations

- Most training data comes from northern Italy (AerialWaste). Almeria looks different, and that is why steps 3-4 exist.
- Plastic is a rare type in AerialWaste: the original paper counts 126 labelled images, and notebook 01 prints the exact number for v3. Plastic scores are less reliable than the waste / no-waste score until Almeria plastic labels are added.
- The classifier works on 210 m windows. Use the detector boxes and the attention maps to point at the exact spot.
- The output is a list of candidate sites to check, not proof of illegal dumping.
