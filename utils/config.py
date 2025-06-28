# utils/config.py  
#Create the utils folder and a config.py file within the folder
#Define the contents of the file as below
#This important utility will be used to load the OpenAI API Key from the .env in all the files we are going to use.
#Ensure the code is properly indented
import os
from dotenv import load_dotenv
def get_openai_config():
    """Load OpenAI configuration from environment variables."""
    load_dotenv()
    return {
    "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    "api_key": os.environ.get("OPENAI_API_KEY"),
    }
