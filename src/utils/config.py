import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def get_client():
    """
    Returns the OpenAI client instance.
    """
    return client