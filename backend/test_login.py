import requests

url = "http://127.0.0.1:8000/login"
data = {"username": "ali", "password": "secret123"}  # same user you registered

r = requests.post(url, data=data)
print("Status:", r.status_code)
print("Text:", r.text)