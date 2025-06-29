import os
import tempfile
import git as git
import shutil

def clone_repo_to_temp(repo_url, username=None, password=None):
    # Create a temporary directory, deleting it if it already exists
    temp_dir = tempfile.mkdtemp()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        temp_dir = tempfile.mkdtemp()
    print(f"Cloning repository into temporary directory: {temp_dir}")
    
    # Add authentication to the repository URL if username and password are provided
    if username and password:
        auth_repo_url = repo_url.replace("https://", f"https://{username}:{password}@")
    else:
        auth_repo_url = repo_url

    # Clone the repository
    try:
        git.Repo.clone_from(auth_repo_url, temp_dir)
        print("Repository cloned successfully.")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")
        return None
    
    return temp_dir

# Example usage
if __name__ == "__main__":
    repository_url = "https://github.com/premrpk/python_repo.git"  # Replace with the actual repository URL
    username = input("Enter your git email id: ")  
    password = input("Enter your git password: ")  
    temp_directory = clone_repo_to_temp(repository_url, username, password)
    if temp_directory:
        print(f"Repository is stored in: {temp_directory}")