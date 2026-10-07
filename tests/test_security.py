import logging
import os
import stat

import pytest

from app import auth, db


# ── JWT signing key ──────────────────────────────────────────────────────────

def test_env_key_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GENAI_SECRET_KEY", "x" * 40)
    assert auth.load_secret_key(tmp_path) == "x" * 40
    assert not (tmp_path / ".jwt_secret").exists()


def test_short_env_key_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("GENAI_SECRET_KEY", "short")
    k = auth.load_secret_key(tmp_path)
    assert k != "short" and len(k) >= auth.MIN_SECRET_LEN


def test_key_is_generated_once_and_persisted(tmp_path, monkeypatch):
    monkeypatch.delenv("GENAI_SECRET_KEY", raising=False)
    first = auth.load_secret_key(tmp_path)
    second = auth.load_secret_key(tmp_path)
    assert first == second and len(first) >= auth.MIN_SECRET_LEN
    assert (tmp_path / ".jwt_secret").read_text().strip() == first
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / ".jwt_secret").stat().st_mode) == 0o600


def test_old_hardcoded_key_is_gone():
    assert auth.SECRET_KEY != "genai-secret-change-me-in-production-2026"
    assert "change-me" not in auth.SECRET_KEY


def test_key_in_unwritable_dir_still_returns_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GENAI_SECRET_KEY", raising=False)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    k = auth.load_secret_key(blocker / "data")   # cannot create a dir under a file
    assert len(k) >= auth.MIN_SECRET_LEN


# ── admin bootstrap ──────────────────────────────────────────────────────────

@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.delenv("GENAI_ADMIN_PASSWORD", raising=False)
    return tmp_path


def test_first_run_admin_gets_a_random_password(empty_db, capsys):
    db.ensure_admin()
    out = capsys.readouterr().out
    pw = next(l.split(":", 1)[1].strip() for l in out.splitlines() if "One-time password" in l)
    assert pw != db.LEGACY_ADMIN_PASSWORD and len(pw) >= 12
    user = db.get_user_by_email(db.DEFAULT_ADMIN_EMAIL)
    assert user["role"] == "admin"
    assert auth.verify_password(pw, user["password_hash"])
    assert not auth.verify_password("admin123", user["password_hash"])
    # the password is not written to disk in clear text
    assert pw not in (empty_db / "users.json").read_text()


def test_two_fresh_installs_get_different_passwords(empty_db, capsys, tmp_path, monkeypatch):
    db.ensure_admin()
    a = capsys.readouterr().out
    monkeypatch.setattr(db, "USERS_FILE", tmp_path / "other.json")
    db.ensure_admin()
    b = capsys.readouterr().out
    assert a != b


def test_admin_password_from_env(empty_db, monkeypatch, capsys):
    monkeypatch.setenv("GENAI_ADMIN_PASSWORD", "correct-horse-battery")
    db.ensure_admin()
    out = capsys.readouterr().out
    assert "correct-horse-battery" not in out          # never echoed
    user = db.get_user_by_email(db.DEFAULT_ADMIN_EMAIL)
    assert auth.verify_password("correct-horse-battery", user["password_hash"])


def test_existing_install_with_default_password_is_flagged(empty_db, caplog):
    db.create_user(db.DEFAULT_ADMIN_EMAIL, auth.hash_password("admin123"), "Administrator", "admin")
    with caplog.at_level(logging.WARNING, logger="app.db"):
        db.ensure_admin()
    assert any("default password" in r.message for r in caplog.records)


def test_existing_install_with_changed_password_is_quiet(empty_db, caplog):
    db.create_user(db.DEFAULT_ADMIN_EMAIL, auth.hash_password("a-much-better-one"), "Administrator", "admin")
    with caplog.at_level(logging.WARNING, logger="app.db"):
        db.ensure_admin()
    assert not caplog.records
    assert len(db.get_users()) == 1                   # nothing was created


def test_reset_script_changes_password(empty_db, monkeypatch):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "set_admin_password", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "set_admin_password.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    db.create_user(db.DEFAULT_ADMIN_EMAIL, auth.hash_password("admin123"), "Administrator", "admin")
    monkeypatch.setattr("sys.argv", ["x"])
    monkeypatch.setenv("GENAI_NEW_PASSWORD", "a-fresh-long-password")
    assert mod.main() == 0
    assert auth.verify_password("a-fresh-long-password", db.get_user_by_email(db.DEFAULT_ADMIN_EMAIL)["password_hash"])
    monkeypatch.setenv("GENAI_NEW_PASSWORD", "admin123")
    assert mod.main() == 1                            # refuses the well-known default
    monkeypatch.setenv("GENAI_NEW_PASSWORD", "short")
    assert mod.main() == 1
    monkeypatch.setattr("sys.argv", ["x", "nobody@example.com"])
    monkeypatch.setenv("GENAI_NEW_PASSWORD", "a-fresh-long-password")
    assert mod.main() == 1
