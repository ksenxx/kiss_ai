# This script will interact with the Langchain API.

import os
import requests

# Set the Langchain API key.
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

# Set the Langchain API base URL.
LANGCHAIN_API_BASE_URL = "https://api.host.langchain.com"

# Set the headers for the API request.
headers = {
    "X-Api-Key": LANGCHAIN_API_KEY,
    "Content-Type": "application/json",
}

def list_deployments():
    """List all deployments."""
    response = requests.get(
        f"{LANGCHAIN_API_BASE_URL}/v2/deployments",
        headers=headers,
    )
    return response.json()

if __name__ == "__main__":
    # Get the Langchain API key from the environment variables.
    if not LANGCHAIN_API_KEY:
        raise ValueError("LANGCHAIN_API_KEY environment variable not set.")

    # List all deployments.
    deployments = list_deployments()
    print(deployments)
