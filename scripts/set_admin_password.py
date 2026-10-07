#!/usr/bin/env python3
"""Set (or reset) a user's password from the command line.

    python scripts/set_admin_password.py                    # admin@example.com, prompts
    python scripts/set_admin_password.py someone@corp.com   # another account
    GENAI_NEW_PASSWORD=... python scripts/set_admin_password.py   # non-interactive

Use this if the one-time first-run password was lost, or to replace the
default password on an older install. Run it on the genai host; it edits
data/users.json directly, so stop-start is not required.
"""
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import auth, db  # noqa: E402


def main() -> int:
    email = (sys.argv[1] if len(sys.argv) > 1 else db.DEFAULT_ADMIN_EMAIL).strip().lower()
    user = db.get_user_by_email(email)
    if not user:
        print(f"No user with email {email}", file=sys.stderr)
        return 1
    pw = os.environ.get("GENAI_NEW_PASSWORD", "")
    if not pw:
        pw = getpass.getpass(f"New password for {email}: ")
        if pw != getpass.getpass("Repeat: "):
            print("Passwords do not match", file=sys.stderr)
            return 1
    if len(pw) < 10:
        print("Use at least 10 characters", file=sys.stderr)
        return 1
    if pw == db.LEGACY_ADMIN_PASSWORD:
        print("That is the well-known default; choose another", file=sys.stderr)
        return 1
    if not db.set_user_password(user["id"], auth.hash_password(pw)):
        print("Could not update the user", file=sys.stderr)
        return 1
    print(f"Password updated for {email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
