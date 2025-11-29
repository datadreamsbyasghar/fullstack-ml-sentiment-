import requests

# Paste the full token you got from test_login.py
token ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGkiLCJleHAiOjE3NjQyNjA4MjJ9.1ymBeEy0FU2yql8jW0as9qPYEV7MNTMl2SXdiRUmBa0" # full token

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

url = "http://127.0.0.1:8000/predict"
payload = {"text": "This movie was amazing!"}

r = requests.post(url, json=payload, headers=headers)
print("Status:", r.status_code)
print("Text:", r.text)