import httpx

c = httpx.Client(base_url="http://127.0.0.1:8102", timeout=20)
print("health", c.get("/health").status_code)
for m in c.get("/v1/models").json()["data"]:
    if m.get("kind") == "audio":
        print(f"  {m['id']:<36} installed={m.get('installed')}")
