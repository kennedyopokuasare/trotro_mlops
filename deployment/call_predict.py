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

print(response.json())
