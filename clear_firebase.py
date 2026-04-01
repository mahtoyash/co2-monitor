import requests
import sys

db_url = "https://co2-monitor-effff-default-rtdb.asia-southeast1.firebasedatabase.app/model_weights.json"

print(f"Attempting to clear: {db_url}")
response = requests.delete(db_url)

if response.status_code == 200:
    print("SUCCESS: Cleared old model weights from Firebase!")
else:
    print(f"FAILED (Status {response.status_code}): {response.text}")
    sys.exit(1)
