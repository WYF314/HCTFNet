import requests
import json

def get_json_response(url, params=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
    return None

url = "https://api.example.com/data"
params = {'param1': 'value1', 'param2': 'value2'}
headers = {'Authorization': 'Bearer your_token'}

json_data = get_json_response(url, params, headers)
if json_data:
    print(json.dumps(json_data, indent=4))
else:
    print("Failed to retrieve data")