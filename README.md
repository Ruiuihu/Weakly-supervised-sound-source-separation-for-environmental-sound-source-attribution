# Recording-level WSCL

Weakly supervised environmental sound localization. Training uses recording-level class tags from a CSV. Time-frequency boxes are not required and are not read.

The detector follows [wetectron](https://github.com/NVlabs/wetectron) and [OD-WSCL](https://github.com/jinhseo/OD-WSCL) (MIL, iterative refinement, source discovery, and weakly supervised contrastive learning), with a spectral energy prior on log-Mel spectrograms.

## Setup

```bash
conda create -n wscl python=3.7
conda activate wscl
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
python setup.py build develop
```

`apex` is optional. The CUDA ops fall back to a plain wrapper when it is not installed.

## Labels

Spectrograms live in class-combination folders, for example `birds_chicken/birds_chicken0.jpg` and `birds_construction_dog/birds_construction_dog_0.jpg`. The folder name is the recording-level tag. Build the CSV with:

```bash
python build_recording_csv.py \
  --image-root /path/to/spec_img_3all \
  --output /path/to/spec_img_3all_labels.csv
```

The CSV has columns `relative_path,classes`. Classes are semicolon-separated and must be one of `birds`, `chicken`, `construction`, `dog`, `flow`, `frog`, `insect`, `rain`.

## Train

```bash
python train_WSCL.py \
  --image-root /path/to/spec_img_3all \
  --label-csv /path/to/spec_img_3all_labels.csv \
  --epochs 50 \
  --save-dir checkpoints
```

Every image is used for training. There is no validation split and no per-epoch mAP, because box annotations are not available yet. Each epoch overwrites `checkpoints/last.pth`.

The default profile is recording-level:

- MIL uses the set of class names in each recording
- `PARTIAL_LABELS=none`
- RPN proposals still come from the RPN head and the energy prior
- RPN box loss and click/point loss are off (`MODEL.RPN.SUPERVISE_WITH_BOXES=False`)
- source discovery and WSCL are on (`SOLVER.CONTRA=True`, `OICR_P=0`)
- external pseudo boxes and `--rpn-click-pseudo-box` are ignored

`--allow-box-supervision` turns box and click targets back on. That path still expects box fields in the dataset and is not the default. `--eval-only` exits until box annotations exist.

Backbone initialization uses ImageNet ResNet-50 unless `--no-imagenet-pretrained` is set. COCO RPN weights are optional via `--coco-rpn-weight`.

Spectrogram images can be built with `wav_to_spectrogram.py`.
