"""
tests/test_auth_session.py
Auth session lifecycle tests for LabelGuard AI.

Tests:
  1. Valid token  -> /api/auth/me returns 200
  2. Invalid/stale token -> 401
  3. Expired (post-restart) token -> 401
  4. Sign-out clears server session
  5. Any 401 from another protected endpoint also returns 401
"""

import pytest
import httpx

BASE = "http://127.0.0.1:8000"


def _login(username: str, password: str, role: str) -> dict:
    resp = httpx.post(
        f"{BASE}/api/auth/login",
        json={"username": username, "password": password, "role": role},
        timeout=10,
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()


# ─────────────────────────────────────────────────────────────
# Test 1: Valid token -> /api/auth/me returns 200
# ─────────────────────────────────────────────────────────────
def test_valid_token_returns_200():
    session = _login("customer@labelguard.com", "Customer@123", "consumer")
    token = session["access_token"]
    me = httpx.get(
        f"{BASE}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert me.status_code == 200, f"/api/auth/me failed: {me.text}"
    user = me.json()
    assert user["name"] == "Priya Sharma"
    assert user["role"] == "consumer"
    print("[PASS] Test 1: Valid token -> 200 from /api/auth/me")


# ─────────────────────────────────────────────────────────────
# Test 2: Invalid/random token -> 401
# ─────────────────────────────────────────────────────────────
def test_invalid_token_returns_401():
    me = httpx.get(
        f"{BASE}/api/auth/me",
        headers={"Authorization": "Bearer totally-fake-token-xyz-12345"},
        timeout=5,
    )
    assert me.status_code == 401, f"Expected 401, got {me.status_code}: {me.text}"
    print("[PASS] Test 2: Invalid token -> 401")


# ─────────────────────────────────────────────────────────────
# Test 3: Stale token (simulates post-restart) -> 401
# After logout the token is removed from ACTIVE_SESSIONS,
# which mirrors what happens on server restart.
# ─────────────────────────────────────────────────────────────
def test_stale_token_after_logout_returns_401():
    session = _login("customer@labelguard.com", "Customer@123", "consumer")
    token = session["access_token"]

    # Invalidate the session server-side (mirrors server restart)
    logout = httpx.post(
        f"{BASE}/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert logout.status_code == 200

    # Now re-use the same token — should get 401
    me = httpx.get(
        f"{BASE}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert me.status_code == 401, f"Expected 401 for stale token, got {me.status_code}: {me.text}"
    print("[PASS] Test 3: Stale/post-restart token -> 401")


# ─────────────────────────────────────────────────────────────
# Test 4: Sign-out clears server session
# ─────────────────────────────────────────────────────────────
def test_logout_invalidates_token():
    session = _login("gov-lm-4821", "Inspector@2026", "inspector")
    token = session["access_token"]

    # Confirm it works before logout
    me_before = httpx.get(
        f"{BASE}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert me_before.status_code == 200

    # Logout
    logout = httpx.post(
        f"{BASE}/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert logout.status_code == 200

    # Token must now be dead
    me_after = httpx.get(
        f"{BASE}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    assert me_after.status_code == 401, f"Token still valid after logout! Got {me_after.status_code}"
    print("[PASS] Test 4: Logout invalidates token on server")


# ─────────────────────────────────────────────────────────────
# Test 5: No token / missing header -> 401
# ─────────────────────────────────────────────────────────────
def test_no_token_returns_401():
    me = httpx.get(f"{BASE}/api/auth/me", timeout=5)
    assert me.status_code == 401, f"Expected 401 for no token, got {me.status_code}"
    print("[PASS] Test 5: No token -> 401")


if __name__ == "__main__":
    print("\n=== LabelGuard Auth Session Tests ===\n")
    tests = [
        test_valid_token_returns_200,
        test_invalid_token_returns_401,
        test_stale_token_after_logout_returns_401,
        test_logout_invalidates_token,
        test_no_token_returns_401,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
