"""Download the three raw datasets used by the custom Mini-VTAB-3 suite."""

from pathlib import Path

from torchvision.datasets import DTD, EuroSAT
from torchvision.datasets.utils import download_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DSPRITES_MD5 = "7da33b31b13a06f4b04a70402ce90c2e"


def download_dtd() -> None:
    root = DATA_DIR / "dtd"
    DTD(root=root, split="train", download=True)
    print(f"DTD downloaded to: {root}")


def download_eurosat() -> None:
    root = DATA_DIR / "eurosat"
    EuroSAT(root=root, download=True)
    print(f"EuroSAT downloaded to: {root}")


def download_dsprites() -> None:
    root = DATA_DIR / "dsprites"
    download_url(
        url=DSPRITES_URL,
        root=root,
        filename=DSPRITES_FILENAME,
        md5=DSPRITES_MD5,
    )
    print(f"dSprites downloaded to: {root}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_dtd()
    download_eurosat()
    download_dsprites()
    print("Mini-VTAB-3 raw datasets are ready.")


if __name__ == "__main__":
    main()
