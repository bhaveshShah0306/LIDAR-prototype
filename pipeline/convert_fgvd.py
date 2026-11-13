"""Convert FGVD (Pascal VOC XML) to YOLO labels in our 7 classes, plus pseudo-labelled people/animals."""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

from extract_frames import CLASSES

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data" / "fgvd", ROOT / "data" / "labels" / "fgvd"
FGVD_TO_OURS = {
    "car": "car",
    "motorcycle": "two_wheeler", "scooter": "two_wheeler",
    "autorickshaw": "auto_rickshaw",
    "truck": "bus_truck", "bus": "bus_truck", "mini-bus": "bus_truck",
}          # from step 2
PSEUDO = {0: "person", 16: "animal", 17: "animal", 18: "animal", 19: "animal"}  # COCO ids
PSEUDO_CONF = 0.5


def vehicle_lines(xml_path):
        root = ET.parse(xml_path).getroot()
        img_w = float(root.find("size/width").text)
        img_h = float(root.find("size/height").text)  # same idea as img_w
        lines = []
        for obj in root.iter("object"):
            vtype = obj.find("name").text.split("_")[0]  # "scooter_Honda_Activa" -> "scooter"
            cls = CLASSES.index(FGVD_TO_OURS.get(vtype, "other"))
            box = obj.find("bndbox")
            x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            cx = (x1 + x2) / (2 * img_w)     # centre x, as a fraction of the image width
            cy = (y1 + y2) / (2 * img_h)     # centre y, as a fraction of the image height
            w = (x2 - x1) / img_w      # box width, as a fraction of the image width
            h = (y2 - y1) / img_h      # box height, as a fraction of the image height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines


def pseudo_lines(model, img_path):
    r = model.predict(str(img_path), conf=PSEUDO_CONF, classes=list(PSEUDO), verbose=False)[0]
    lines = []
    for (cx, cy, w, h), k in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
        name = PSEUDO[int(k)]
        cls = CLASSES.index(name)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert(split, model):
    """data/fgvd/<split>/images + annos -> data/labels/fgvd/<split>/images + labels."""
    img_out = DST / split / "images"
    lbl_out = DST / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted((SRC / split / "annos").glob("*.xml"))
    n_pseudo = 0
    for i, xml_path in enumerate(xml_files, start=1):
        img = SRC / split / "images" / (xml_path.stem + ".jpg")
        if not img.exists():
            print("No image for", xml_path.name)
            continue
        lines = vehicle_lines(xml_path)
        extra = pseudo_lines(model, img)                           # people/animals for this image
        n_pseudo += len(extra)
        shutil.copy(img, img_out)                 # copy into img_out, keeping the same file name
        (lbl_out / (xml_path.stem + ".txt")).write_text("\n".join(lines + extra) + "\n")   # vehicles + extra
        if i % 200 == 0:
            print(f"{split}: {i}/{len(xml_files)}")
    print(f"{split}: {len(xml_files)} images, {n_pseudo} pseudo-labelled people/animals")


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    for split in ("train", "val"):
        convert(split, model)