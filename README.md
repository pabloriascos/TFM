# TFM

This repository contains scripts to train a YOLOv8 model and perform object
tracking using Deep SORT. The provided `best.pt` model is loaded by
`main_tracking.py` to detect pigs in video files.

## Setup

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Run the training script to train or fine tune the YOLOv8 model:

```bash
python train.py
```

## Tracking

Use `main_tracking.py` to run object tracking on a video. The script expects the
input video at `./data/dos_1.mp4` and writes the output video and tracking data
to the project directory.

```bash
python main_tracking.py
```
