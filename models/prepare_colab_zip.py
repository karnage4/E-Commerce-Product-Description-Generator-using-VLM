"""
Dataset upload helper — run this LOCALLY before going to Colab.

Creates a compact zip of only the files needed for Colab training:
  - data/processed/metadata/listings_final.jsonl
  - data/processed/splits/*.txt
  - data/processed/images/**/*.jpg  (all product images)

Output: daraz_dataset_colab.zip  (~varies based on image count)

Then upload this zip to Google Drive for use in Colab.
"""

import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.shared.config import METADATA_FILE, IMAGES_DIR, SPLITS_DIR, PROJECT_ROOT


def create_colab_zip(output_name: str = "daraz_dataset_colab.zip") -> None:
    output_path = PROJECT_ROOT / output_name
    processed   = IMAGES_DIR.parent   # data/processed/

    files_to_zip: list[Path] = []

    # Metadata
    if METADATA_FILE.exists():
        files_to_zip.append(METADATA_FILE)
    else:
        print(f"[!] Missing: {METADATA_FILE}")

    # Splits
    for split_file in SPLITS_DIR.glob("*.txt"):
        files_to_zip.append(split_file)

    # Images
    image_files = list(IMAGES_DIR.rglob("*.jpg"))
    print(f"  Found {len(image_files)} product images")
    files_to_zip.extend(image_files)

    total_mb = sum(f.stat().st_size for f in files_to_zip if f.exists()) / (1024 ** 2)
    print(f"  Total size: {total_mb:.0f} MB → zipping to {output_path}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in tqdm(files_to_zip, desc="Zipping", unit="file"):
            if not file_path.exists():
                continue
            # Archive name: relative to data/processed/
            arcname = file_path.relative_to(processed)
            zf.write(file_path, arcname)

    zip_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n  Done! {output_path}  ({zip_mb:.0f} MB)")
    print(f"\n  Next steps:")
    print(f"    1. Upload '{output_name}' to Google Drive")
    print(f"    2. Open Google Colab: https://colab.research.google.com/")
    print(f"    3. Follow the guide in models/COLAB_GUIDE.md")


if __name__ == "__main__":
    create_colab_zip()
