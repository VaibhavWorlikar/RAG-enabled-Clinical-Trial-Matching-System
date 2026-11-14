"""
End-to-end smoke test against a running API (local or deployed).
Uses real OpenAI calls and the real database. Not collected by pytest.

    python tests/e2e_smoke.py                                   # http://127.0.0.1:8000, all scenarios
    python tests/e2e_smoke.py crohn https://your-app.vercel.app
"""
import sys
import time
import uuid

import httpx

WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
BASE = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000").rstrip("/") + "/api"
c = httpx.Client(timeout=180)


def register(name: str, role: str = "patient") -> httpx.Response:
    return c.post(f"{BASE}/auth/register", json={"username": name, "password": "testpass123", "role": role})


def chat(tok: str, msg: str, sid: str | None = None) -> dict:
    t = time.time()
    r = c.post(f"{BASE}/chat", json={"message": msg, "session_id": sid, "stream": False},
               headers={"Authorization": f"Bearer {tok}"})
    r.raise_for_status()
    d = r.json()
    print(f"\n>>> {msg}  [{time.time() - t:.1f}s]")
    print("profile:", {k: v for k, v in d["patient_profile"].items() if v})
    print(f"flags: clarify={d['clarification_needed']} gaps={d['awaiting_gap_answers']} found={d['trials_found']}")
    for tr in d["trials"]:
        print(f"  [{tr['eligibility']:8}] {tr['trial_id']} {tr['title'][:90]}")
    print("reply:", d["reply"][:900].replace("\n", " | "))
    return d


user = "t" + uuid.uuid4().hex[:6]
print("admin self-register (expect 422) ->", register(user + "a", "admin").status_code)
tok = register(user).json()["access_token"]

if WHICH in ("all", "crohn"):
    d = chat(tok, "My doctor thinks I have Crohn's disease. I'm 34, male, not on medication yet.")
    if d["awaiting_gap_answers"]:
        chat(tok, "Diagnosis was confirmed by colonoscopy last month. I've only tried mesalamine. "
                  "I don't smoke and have no other conditions.", d["session_id"])
if WHICH in ("all", "diabetes"):
    d = chat(tok, "I'm a 45-year-old man with type 2 diabetes and high blood pressure, on metformin.")
    if d["awaiting_gap_answers"]:
        chat(tok, "skip", d["session_id"])
if WHICH in ("all", "clarify"):
    d = chat(tok, "I have asthma")
    chat(tok, "I'm 29 and female", d["session_id"])
if WHICH in ("all", "isolation"):
    d = chat(tok, "I'm 60, female, with breast cancer")
    tok2 = register(user + "b").json()["access_token"]
    r = c.get(f"{BASE}/chat/sessions/{d['session_id']}", headers={"Authorization": f"Bearer {tok2}"})
    print("\nother user reading session (expect 404) ->", r.status_code)
    r = c.get(f"{BASE}/chat/sessions/{d['session_id']}", headers={"Authorization": f"Bearer {tok}"})
    print("owner reading session (expect 200) ->", r.status_code)
