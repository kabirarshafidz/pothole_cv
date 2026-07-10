import argparse

from utils import COUNTRIES, DATA_ROOT, SPLITS_ROOT, country_of

SPLITS = ("train", "val", "test")


def build_lists() -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        image_dir = DATA_ROOT / split / "images"
        by_country = {country: [] for country in COUNTRIES}
        for img_path in sorted(image_dir.glob("*.jpg")):
            country = country_of(img_path.name)
            if country:
                by_country[country].append(img_path)

        for country, paths in by_country.items():
            out_path = SPLITS_ROOT / f"{country}_{split}.txt"
            out_path.write_text("\n".join(str(p) for p in paths))
            print(f"{country} {split}: {len(paths)} images -> {out_path}")


def main() -> None:
    argparse.ArgumentParser(
        description="Build per-country image list files from the pre-split RDD_SPLIT dataset"
    ).parse_args()
    build_lists()


if __name__ == "__main__":
    main()
