"""Create .env from .env.example with freshly generated secrets (standard library only - no install needed).

    python scripts/init_env.py

An existing .env is kept: only secrets that are empty or still placeholders are filled in.
"""
from __future__ import annotations

import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ("API_KEY", "MAILGUN_SIGNING_KEY", "WEBHOOK_SHARED_SECRET")


def main() -> None:
    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8-sig") if env.exists() else ""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    if not text.strip():
        text = example
    else:  # add settings introduced since the file was created
        have = set(re.findall(r"^([A-Z_]+)=", text, re.M))
        new = [line for line in example.splitlines() if re.match(r"[A-Z_]+=", line) and line.split("=")[0] not in have]
        if new:
            text = text.rstrip("\n") + "\n\n# --- added by init_env.py ---\n" + "\n".join(new) + "\n"
    for name in GENERATED:
        text = re.sub(rf"^{name}=(?:change-me\S*)?[ \t]*$", f"{name}={secrets.token_urlsafe(24)}", text, flags=re.M)
    env.write_text(text, encoding="utf-8", newline="\n")
    print(f"{env} is ready.")
    if re.search(r"^LLM_PROVIDER=anthropic\s*$", text, re.M) and re.search(r"^ANTHROPIC_API_KEY=\s*$", text, re.M):
        print("Next: put your key after ANTHROPIC_API_KEY= (or set LLM_PROVIDER=heuristic to run offline).")


if __name__ == "__main__":
    main()
