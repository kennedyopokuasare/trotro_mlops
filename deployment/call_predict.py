import json

import requests

trotro = {
    "tpep_pickup_datetime": "2021-01-01 00:15:56",
    "PULocationID": 10,
    "DOLocationID": 50,
    "RatecodeID": 1,
    "trip_distance": 40.0,
    "passenger_count": 1.0,
}


URL = "http://localhost:9696/predict"

response = requests.post(url=URL, json=trotro, timeout=5)
prediction = response.json()

print("Prediction:")
print(json.dumps(prediction, indent=2))

print("Asserting that prediction contains duration")
assert "duration" in prediction

print("Asserting that predicted duration is not None")
assert prediction["duration"] is not None
