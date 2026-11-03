"""Fine-tune YOLO on your labelled frames (AI #1). Run this on Kaggle with a GPU, not the laptop.

Supervised learning, specifically TRANSFER LEARNING: start from weights that already know
general shapes (trained on COCO), and teach them your classes and your camera view.

Kaggle setup:
    1. Zip data/labels (train/ and test/ folders + classes.txt), upload as a PRIVATE dataset
       (the frames show real people and number plates).
    2. New notebook -> Settings: Accelerator = GPU, Internet = on. Add your dataset.
    3. !pip install ultralytics   then copy this file's code into cells, or upload and run it.
    4. In train/data.yaml set `path:` to /kaggle/input/<your-dataset-name>

Output: runs/detect/<name>/weights/best.pt   -> download it into models/ on the laptop
        results/detector.md                  -> before/after table (you write it from the numbers)

Done when: you have mAP50 and per-class recall for BOTH the pretrained model and yours,
on the same 50 test frames.

Concepts to understand (write the answers in NOTES.md):
    - epoch, batch size, learning rate, overfitting
    - what mAP50 and mAP50-95 measure, and precision vs recall
    - why the test frames come from a ride that has NO frames in the training set
"""
import argparse

from ultralytics import YOLO


def train(data, base="yolo11n.pt", epochs=50, imgsz=640, batch=16, name="rear_alert"):
    """Fine-tune and return the path to best.pt.

    TODO:
      1. model = YOLO(base)
      2. model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, name=name, patience=15)
         `patience` stops early if the validation score hasn't improved for 15 epochs.
      3. Return the best weights path: model.trainer.best

    Check yourself: training loss keeps falling but validation mAP stops improving after
    epoch 30. What is happening, and what does `patience` do about it?
    """
    raise NotImplementedError("TODO: train")


def evaluate(weights, data, imgsz=640):
    """Score a model on the val (test) split. Return a dict of the numbers you'll report.

    TODO:
      1. metrics = YOLO(weights).val(data=data, imgsz=imgsz, split="val")
      2. Collect: metrics.box.map50, metrics.box.map, and per-class recall
         (metrics.box.r is one value per class, in the order of metrics.names).
      3. Return them in a dict, e.g. {"mAP50": ..., "mAP50-95": ..., "recall": {"car": ..., ...}}

    Fair comparison: the PRETRAINED model doesn't know your class numbers (it uses COCO's 80).
    Simplest honest approach: compare on the classes both share (car, bus_truck, two_wheeler,
    person, animal), and report auto_rickshaw / other as "pretrained: no such class".
    """
    raise NotImplementedError("TODO: evaluate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="train/data.yaml")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    best = train(args.data, args.base, args.epochs)
    print("Fine-tuned:", evaluate(best, args.data))
