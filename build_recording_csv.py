# -*- coding: utf-8 -*-
"""Build a recording-level label CSV from spectrogram folder names.

Folder names are underscore-separated class tokens, for example
``birds_chicken`` and ``birds_construction_dog``. Image filenames append an
index and are not parsed.
"""

import argparse
import csv
import os

KNOWN_CLASSES = (
    "birds",
    "chicken",
    "construction",
    "dog",
    "flow",
    "frog",
    "insect",
    "rain",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_ROOT = r"D:\sound_segments\sound_mixed_paper\code\spec_img_3all"
DEFAULT_OUTPUT = r"D:\sound_segments\sound_mixed_paper\code\spec_img_3all_labels.csv"


def classes_from_folder(folder_name, known_classes):
    parts = [part.strip().lower() for part in folder_name.split("_") if part.strip()]
    unknown = [part for part in parts if part not in known_classes]
    if not parts or unknown:
        raise ValueError(
            "Folder %r is not a list of known classes %s. Unknown tokens: %s"
            % (folder_name, ",".join(known_classes), ",".join(unknown) or "(empty)")
        )
    return parts


def collect_rows(image_root, known_classes):
    image_root = os.path.normpath(image_root)
    rows = []
    for dirpath, _, filenames in os.walk(image_root):
        folder_name = os.path.basename(dirpath)
        if os.path.normcase(dirpath) == os.path.normcase(image_root):
            continue
        image_names = [
            name
            for name in filenames
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not image_names:
            continue
        classes = classes_from_folder(folder_name, known_classes)
        class_field = ";".join(classes)
        for name in sorted(image_names):
            rel = os.path.relpath(os.path.join(dirpath, name), image_root)
            rows.append((rel.replace("\\", "/"), class_field))
    rows.sort(key=lambda item: item[0])
    return rows


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "classes"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Write recording-level labels from folder names.")
    parser.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    known = set(KNOWN_CLASSES)
    rows = collect_rows(args.image_root, known)
    if not rows:
        raise SystemExit("No images found under %s" % args.image_root)
    write_csv(rows, args.output)
    print("Wrote %d rows to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
