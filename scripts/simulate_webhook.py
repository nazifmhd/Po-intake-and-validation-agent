"""Post a correctly *signed* Mailgun-style inbound email to the running API - no domain or tunnel needed.

    python scripts/simulate_webhook.py data/samples/01_clean_northwind.pdf [more files] \
        [--url http://localhost:8000] [--provider mailgun|generic] [--message-id <id>] [--bad-signature]

--bad-signature shows the spoofed-post rejection (HTTP 401). Re-sending with the same --message-id shows dedup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.config import get_settings
    from app.intake.simulate import NotConfigured, post_email

    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--provider", default="mailgun", choices=["mailgun", "generic"])
    ap.add_argument("--message-id", default=None)
    ap.add_argument("--sender", default="Dana Buyer <dana@northbridge.example>")
    ap.add_argument("--subject", default="Purchase order attached")
    ap.add_argument("--bad-signature", action="store_true")
    args = ap.parse_args()
    try:
        r = post_email([(Path(f).name, Path(f).read_bytes()) for f in args.files], get_settings(), url=args.url,
                       sender=args.sender, subject=args.subject, provider=args.provider, message_id=args.message_id,
                       bad_signature=args.bad_signature)
    except NotConfigured as exc:
        sys.exit(str(exc))
    print(r.status_code, r.text)


if __name__ == "__main__":
    main()
