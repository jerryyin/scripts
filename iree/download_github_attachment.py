# pip install requests bs4

import os  
import requests  
import sys
from bs4 import BeautifulSoup  

def download_attachments(issue_url, download_dir='downloads'):  
    # Fetch the issue page  
    response = requests.get(issue_url)  
    response.raise_for_status()  # Raise an error for bad responses  

    # Parse the page content  
    soup = BeautifulSoup(response.text, 'html.parser')  

    # Find all attachment links  
    attachment_links = soup.find_all('a', href=True)  
    file_urls = [link['href'] for link in attachment_links if '/user-attachments/' in link['href']]  

    # Create the download directory if it doesn't exist  
    os.makedirs(download_dir, exist_ok=True)  

    # Download each file  
    for file_url in file_urls:  
        file_name = file_url.split('/')[-1]  
        file_path = os.path.join(download_dir, file_name)  
        file_response = requests.get(file_url)  
        file_response.raise_for_status()  

        with open(file_path, 'wb') as file:  
            file.write(file_response.content)  
        print(f'Downloaded: {file_name}')  

if __name__ == '__main__':  
    if len(sys.argv) != 2:
        print("Usage: python download_github_attachments.py <issue_id>")
        sys.exit(1)

    # Get the issue ID from the command-line argument
    issue_id = sys.argv[1]
    repo_owner = 'iree-org'  
    repo_name = 'iree'  
    issue_url = f'https://github.com/{repo_owner}/{repo_name}/issues/{issue_id}'  

    download_attachments(issue_url)  

