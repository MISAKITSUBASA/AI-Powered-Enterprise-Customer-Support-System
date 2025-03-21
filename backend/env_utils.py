import os
from dotenv import load_dotenv

def get_openai_api_key():
    """
    Get the OpenAI API key from environment variables.
    Ensures the key is loaded from .env file through dotenv.
    """
    # Force reload the environment variables from .env to ensure we get the latest values
    load_dotenv(override=True)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")
        
    return api_key

def print_masked_key(api_key, message="API Key"):
    """
    Print a masked version of the API key for debugging purposes.
    Only shows the first 4 and last 4 characters.
    """
    if not api_key:
        print(f"{message}: Not provided")
        return
        
    if len(api_key) <= 8:
        masked = "****"
    else:
        masked = f"{api_key[:4]}...{api_key[-4:]}"
        
    print(f"{message}: {masked}")
