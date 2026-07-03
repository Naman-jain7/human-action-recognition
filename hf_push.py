from huggingface_hub import login, upload_folder
import os
from dotenv import load_dotenv

load_dotenv()

# (optional) Login with your Hugging Face credentials
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

HF_FOLDER_PATH = os.getenv("HF_FOLDER_PATH")
HF_REPO_ID = os.getenv("HF_REPO_ID")

# Push your model files
upload_folder(folder_path=HF_FOLDER_PATH, repo_id=HF_REPO_ID, repo_type="model") # type: ignore
