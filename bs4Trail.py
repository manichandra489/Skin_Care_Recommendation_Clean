import requests
from bs4 import BeautifulSoup

url = "https://www.lorealparisusa.com/skin-care/anti-aging"

# Define custom headers as a dictionary
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}

# Pass the headers dictionary to the requests.get() method
response = requests.get(url, headers=headers)

# Parse the content with Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')
searchResults = soup.find_all('main')
with open('output.html', 'w', encoding='utf-8') as file:
    file.write(str(searchResults))


# Now you can scrape the page content
