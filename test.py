import requests

# Define the base URL of your Hugging Face Space
base_url = "https://huggingface.co/spaces/deFEost23/anomalidet"

# Define the query parameters
params = {
    "request_interval": 15,  # example value in seconds
    "token_length": 7000,    # example token length
    "model_number": 2        # example model number
}

# Send a GET request with the parameters
response = requests.get(base_url, params=params)

# Check if the request was successful and print the response
if response.status_code == 200:
    print("Response from the Space:")
    print(response.text)
else:
    print(f"Request failed with status code {response.status_code}")
