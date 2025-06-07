from huggingface_hub import login, whoami
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = "#token"

try:
    login(HF_TOKEN)
    user_info = whoami()
    print(f"✅ Logged in as: {user_info['name']}")
except HfHubHTTPError as e:
    print(f"❌ Login failed: {e}")
