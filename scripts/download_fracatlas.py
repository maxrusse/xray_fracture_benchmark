from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request
import zipfile


DEFAULT_API_URL = "https://api.figshare.com/v2/articles/22363012"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def md5sum(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: pathlib.Path) -> None:
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        next_report = 10
        with target.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct >= next_report:
                        print(f"  download: {pct}% ({downloaded}/{total} bytes)")
                        next_report += 10
    print(f"download complete: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FracAtlas from figshare.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--extract-dir", default="data/raw/fracatlas")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    extract_dir = pathlib.Path(args.extract_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching metadata: {args.api_url}")
    metadata = fetch_json(args.api_url)
    files = metadata.get("files", [])
    if not files:
        raise RuntimeError("No files available in figshare metadata.")

    archive = next((f for f in files if f.get("name", "").lower().endswith(".zip")), files[0])
    archive_name = archive["name"]
    archive_url = archive["download_url"]
    expected_md5 = archive.get("supplied_md5", "").lower()

    archive_path = output_dir / archive_name
    if archive_path.exists() and not args.force_download:
        print(f"archive exists, skipping download: {archive_path}")
    else:
        print(f"downloading: {archive_url}")
        download_file(archive_url, archive_path)

    if expected_md5:
        actual_md5 = md5sum(archive_path)
        print(f"md5 expected={expected_md5} actual={actual_md5}")
        if actual_md5 != expected_md5:
            raise RuntimeError("MD5 checksum mismatch for downloaded archive.")

    if args.skip_extract:
        print("skip-extract set, done.")
        return 0

    marker = extract_dir / "FracAtlas" / "dataset.csv"
    if marker.exists() and not args.force_extract:
        print(f"dataset already extracted, skipping: {marker}")
        return 0

    print(f"extracting {archive_path} -> {extract_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    print("extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
