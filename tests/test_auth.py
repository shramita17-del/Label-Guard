import unittest
from fastapi.testclient import TestClient
from backend.main import app, ACTIVE_SESSIONS

class TestAuthenticationFlows(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        ACTIVE_SESSIONS.clear()

    def test_consumer_login_valid(self):
        payload = {
            "role": "consumer",
            "username": "customer@labelguard.com",
            "password": "Customer@123",
            "remember_me": True
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["role"], "consumer")
        self.assertEqual(data["user"]["name"], "Priya Sharma")
        self.assertEqual(data["user"]["id"], "CUST-8831")

    def test_inspector_login_valid(self):
        payload = {
            "role": "inspector",
            "username": "GOV-LM-4821",
            "password": "Inspector@2026",
            "remember_me": False
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["role"], "inspector")
        self.assertEqual(data["user"]["name"], "Insp. Rajesh Kumar")
        self.assertEqual(data["user"]["id"], "GOV-LM-4821")

    def test_login_case_insensitivity(self):
        payload = {
            "role": "inspector",
            "username": "gov-lm-4821",
            "password": "Inspector@2026"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 200)

    def test_login_invalid_password(self):
        payload = {
            "role": "consumer",
            "username": "customer@labelguard.com",
            "password": "WrongPassword!"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid credentials", res.json()["detail"])

    def test_login_invalid_username(self):
        payload = {
            "role": "consumer",
            "username": "unknown@example.com",
            "password": "Customer@123"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid credentials", res.json()["detail"])

    def test_login_cross_role_rejection(self):
        # Attempting to log in as consumer using an Inspector's ID
        payload = {
            "role": "consumer",
            "username": "GOV-LM-4821",
            "password": "Inspector@2026"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access Denied", res.json()["detail"])
        self.assertIn("inspector", res.json()["detail"])

    def test_login_method_not_allowed_on_get(self):
        res = self.client.get("/api/auth/login")
        self.assertEqual(res.status_code, 405)

    def test_otp_request_and_verify_success(self):
        req_res = self.client.post("/api/auth/otp/request", json={
            "identifier": "customer@labelguard.com",
            "role": "consumer"
        })
        self.assertEqual(req_res.status_code, 200)
        otp = req_res.json().get("demo_otp")
        self.assertEqual(otp, "482190")

        # Verify OTP
        verify_res = self.client.post("/api/auth/otp/verify", json={
            "identifier": "customer@labelguard.com",
            "otp": otp,
            "role": "consumer"
        })
        self.assertEqual(verify_res.status_code, 200)
        data = verify_res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["role"], "consumer")

    def test_otp_verify_invalid_code(self):
        verify_res = self.client.post("/api/auth/otp/verify", json={
            "identifier": "customer@labelguard.com",
            "otp": "000000",
            "role": "consumer"
        })
        self.assertEqual(verify_res.status_code, 400)
        self.assertIn("Invalid or expired OTP", verify_res.json()["detail"])

    def test_get_me_and_logout_lifecycle(self):
        # 1. Login
        login_res = self.client.post("/api/auth/login", json={
            "role": "inspector",
            "username": "GOV-LM-4821",
            "password": "Inspector@2026"
        })
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get profile
        me_res = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["name"], "Insp. Rajesh Kumar")

        # 3. Logout
        logout_res = self.client.post("/api/auth/logout", headers=headers)
        self.assertEqual(logout_res.status_code, 200)

        # 4. Subsequent me request must fail with 401
        expired_res = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(expired_res.status_code, 401)

    def test_unauthenticated_me_access(self):
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

if __name__ == '__main__':
    unittest.main()
