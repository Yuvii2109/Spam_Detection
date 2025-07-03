import nltk
import os

# Download the required NLTK resources to your project folder
TARGET_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")

nltk.data.path.append(TARGET_DIR)

nltk.download('punkt', download_dir=TARGET_DIR)
nltk.download('punkt_tab', download_dir=TARGET_DIR)
nltk.download('stopwords', download_dir=TARGET_DIR)

print("✅ All resources downloaded to:", TARGET_DIR)