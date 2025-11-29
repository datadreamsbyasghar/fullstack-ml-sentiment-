import requests

url = "http://127.0.0.1:8000/register"
payload = {"username": "ali", "password": "secret123"}

r = requests.post(url, json=payload)

print("Status:", r.status_code)
print("Text:", r.text)   # raw response