import requests

# Backend URL
url = "http://127.0.0.1:8000/predict"

# Example texts
examples = [
    "This movie was amazing!",
    "I hated every minute of it.",
    "The plot was weak but the acting was great."
]

for text in examples:
    response = requests.post(url, json={"text": text})
    print(f"Input: {text}")
    print("Response:", response.json())
    print("-" * 40)