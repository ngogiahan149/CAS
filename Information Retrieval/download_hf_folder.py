import argparse
from huggingface_hub import snapshot_download

def download_folder(repo_id, folder_path, local_dir="./"):
    snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=[f"{folder_path}/*"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specific folder from a Hugging Face repository.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID on Hugging Face")
    parser.add_argument("--folder_path", type=str, required=True, help="Path of the folder to download")
    parser.add_argument("--local_dir", type=str, default="./", help="Local directory to save the folder")
    
    args = parser.parse_args()

    download_folder(args.repo_id, args.folder_path, args.local_dir)
