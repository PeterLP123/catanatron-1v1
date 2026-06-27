#!/usr/bin/env python3
"""
Sync Colonist training artifacts with the project S3 bucket.

Requires: pip install boto3
Env:      CATANATRON_S3_BUCKET, optional AWS_DEFAULT_REGION
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bucket(explicit: str | None) -> str:
    name = explicit or os.environ.get("CATANATRON_S3_BUCKET", "").strip()
    if not name:
        print(
            "Set CATANATRON_S3_BUCKET or pass --bucket.",
            file=sys.stderr,
        )
        sys.exit(1)
    return name


def _client():
    try:
        import boto3
    except ImportError:
        print("Install boto3: pip install boto3", file=sys.stderr)
        sys.exit(1)
    return boto3.client("s3")


def _run_prefix(run_dir: Path) -> str:
    return f"runs/{run_dir.name}/"


def _upload_tree(s3, bucket: str, local: Path, prefix: str, *, dry_run: bool) -> int:
    count = 0
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local).as_posix()
        if rel.startswith("tb_") or rel.startswith("tensorboard/"):
            continue
        key = f"{prefix}{rel}"
        if dry_run:
            print(f"PUT s3://{bucket}/{key} <- {path}")
        else:
            s3.upload_file(str(path), bucket, key)
        count += 1
    return count


def _download_tree(s3, bucket: str, local: Path, prefix: str, *, dry_run: bool) -> int:
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.startswith(prefix) or key.endswith("/"):
                continue
            rel = key[len(prefix) :]
            dest = local / rel
            if dry_run:
                print(f"GET s3://{bucket}/{key} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
            count += 1
    return count


def cmd_upload_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    bucket = _bucket(args.bucket)
    s3 = _client()
    prefix = _run_prefix(run_dir)
    n = _upload_tree(s3, bucket, run_dir, prefix, dry_run=args.dry_run)
    print(f"{'Would upload' if args.dry_run else 'Uploaded'} {n} files to s3://{bucket}/{prefix}")


def cmd_download_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    bucket = _bucket(args.bucket)
    s3 = _client()
    prefix = _run_prefix(run_dir)
    n = _download_tree(s3, bucket, run_dir, prefix, dry_run=args.dry_run)
    print(f"{'Would download' if args.dry_run else 'Downloaded'} {n} files from s3://{bucket}/{prefix}")


def cmd_list_runs(args: argparse.Namespace) -> None:
    bucket = _bucket(args.bucket)
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix="runs/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp["Prefix"]
            run_id = p.removeprefix("runs/").rstrip("/")
            if run_id:
                seen.add(run_id)
    for run_id in sorted(seen):
        print(run_id)
    if not seen:
        print("(no runs/ prefixes yet)", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", help="Override CATANATRON_S3_BUCKET")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload-run", help="Upload a local runs/<id>/ tree")
    up.add_argument("run_dir", type=Path)
    up.add_argument("--dry-run", action="store_true")
    up.set_defaults(func=cmd_upload_run)

    down = sub.add_parser("download-run", help="Download runs/<id>/ from S3")
    down.add_argument("run_dir", type=Path)
    down.add_argument("--dry-run", action="store_true")
    down.set_defaults(func=cmd_download_run)

    ls = sub.add_parser("list-runs", help="List run IDs under runs/")
    ls.set_defaults(func=cmd_list_runs)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
