import argparse

from utils import COUNTRIES, DATA_ROOT, SPLITS_ROOT, country_of

SPLITS = ("train", "val", "test")


def build_lists() -> None:
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        image_dir = DATA_ROOT / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"{image_dir} does not exist — check DATA_ROOT ({DATA_ROOT})")

        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        if not images:
            raise FileNotFoundError(f"No images found in {image_dir}")

        by_country = {country: [] for country in COUNTRIES}
        unmatched = []
        for img_path in images:
            country = country_of(img_path.name)
            if country:
                by_country[country].append(img_path)
            else:
                unmatched.append(img_path.name)

        if unmatched:
            print(
                f"WARNING: {len(unmatched)} images in {split} matched no country prefix "
                f"in COUNTRIES, e.g. {unmatched[0]!r} — check filename format/casing"
            )

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
