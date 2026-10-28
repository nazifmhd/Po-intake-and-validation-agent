"""Command line:

    python -m app.cli kb build [--if-missing]        build the 3 Chroma collections
    python -m app.cli process FILE [FILE ...]        run files through intake + pipeline (as one "email")
          [--no-rag]                                 ablation: knowledge base off
    python -m app.cli imap [--once]                  IMAP poller
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from app.config import get_settings
from app.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="po-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kb = sub.add_parser("kb")
    kb_sub = kb.add_subparsers(dest="kb_cmd", required=True)
    kb_build = kb_sub.add_parser("build")
    kb_build.add_argument("--if-missing", action="store_true")

    proc = sub.add_parser("process")
    proc.add_argument("files", nargs="+")
    proc.add_argument("--no-rag", action="store_true", help="ablation run with the knowledge base disabled")
    proc.add_argument("--sender", default="cli@local")

    imap = sub.add_parser("imap")
    imap.add_argument("--once", action="store_true")

    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False, stream=sys.stderr)  # stdout = JSON result

    if args.cmd == "kb":
        from app.rag.ingest_kb import build_knowledge_base
        print(json.dumps(build_knowledge_base(settings, if_missing=args.if_missing), indent=2))

    elif args.cmd == "process":
        from app.intake.common import InboundAttachment, InboundEmail, accept_email, run_job
        from app.rag.ingest_kb import build_knowledge_base

        build_knowledge_base(settings, if_missing=True)
        email = InboundEmail(
            message_id=f"<cli-{uuid.uuid4().hex}@po-agent>", sender=args.sender, subject="CLI submission",
            attachments=[InboundAttachment(Path(f).name, Path(f).read_bytes()) for f in args.files], source="cli",
        )
        result = accept_email(email, settings)
        options = {"rag_enabled": False} if args.no_rag else None
        out = result.as_dict()
        out["results"] = [run_job(j, result.message_id, options=options, settings=settings) for j in result.jobs]
        for r in out["results"]:
            r.pop("trace", None)
        print(json.dumps(out, indent=2, default=str))

    elif args.cmd == "imap":
        from app.intake.imap_poller import poll_once
        if args.once:
            print(json.dumps(poll_once(settings), indent=2, default=str))
        else:
            sys.argv = ["imap_poller"]
            from app.intake.imap_poller import main as imap_main
            imap_main()


if __name__ == "__main__":
    main()
