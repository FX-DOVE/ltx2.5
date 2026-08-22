"""
tools/probe_r2_upload.py
─────────────────────────────────────────────────────────────────────────────
Check the object-storage path in `handler._upload_to_s3` without spending GPU
time on a video.

Constructs the boto3 client exactly the way the handler does, so a failure here
is a failure there. Uploads a payload under a `_probe/` key, fetches the
presigned URL back over plain HTTPS the way a browser would, then deletes the
object so nothing is left in the bucket.

    S3_BUCKET=... S3_ENDPOINT_URL=... S3_REGION=auto \
    S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=... \
    python tools/probe_r2_upload.py [path/to/clip.mp4]

Prints only pass/fail — never a credential.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import pathlib
import sys
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError


def _payload(argv: list[str]) -> bytes:
    """The file named on the command line, or a 256 KB filler blob."""
    if len(argv) > 1:
        return pathlib.Path(argv[1]).read_bytes()
    return b"\x00" * (256 * 1024)


def main(argv: list[str]) -> int:
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_KEY_PREFIX", "ltx-2.5")
    ttl = int(os.environ.get("PRESIGNED_URL_TTL_SECONDS", "86400"))
    key = f"{prefix}/_probe/upload-probe.mp4"

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )

    body = _payload(argv)
    print(f"bucket={bucket!r} key={key!r} payload={len(body) / 1e6:.2f} MB")

    # 1. put_object — the call the handler makes.
    try:
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="video/mp4")
    except ClientError as exc:
        error = exc.response.get("Error", {})
        print(f"FAIL put_object: {error.get('Code')} {error.get('Message')}")
        return 1
    except Exception as exc:
        print(f"FAIL put_object: {type(exc).__name__}: {exc}")
        return 1
    print("PASS put_object")

    # 2. generate_presigned_url — R2 is the usual place this misbehaves, since
    #    presigning needs S3_REGION=auto and the account-specific endpoint.
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl
    )
    print(f"PASS generate_presigned_url (host={url.split('/')[2]}, len={len(url)})")

    # 3. Fetch it back unsigned — proves the URL is usable by the caller, not
    #    just well-formed.
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            fetched = response.read()
            ctype = response.headers.get("Content-Type")
        if len(fetched) != len(body):
            print(f"FAIL fetch: {len(fetched)} bytes back, expected {len(body)}")
            return 1
        print(f"PASS fetch presigned URL ({len(fetched) / 1e6:.2f} MB, "
              f"Content-Type={ctype})")
    except urllib.error.HTTPError as exc:
        print(f"FAIL fetch: HTTP {exc.code} {exc.reason}")
        print(exc.read()[:600].decode("utf-8", "replace"))
        return 1
    except Exception as exc:
        print(f"FAIL fetch: {type(exc).__name__}: {exc}")
        return 1

    # 4. Clean up so the probe leaves nothing in the production bucket.
    try:
        client.delete_object(Bucket=bucket, Key=key)
        print("PASS delete_object (probe object removed)")
    except Exception as exc:
        print(f"WARN delete_object failed, {key} left behind: {exc}")

    print("\nObject-storage path is working end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
