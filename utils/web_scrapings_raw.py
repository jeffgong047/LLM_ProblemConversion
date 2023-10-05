import requests
from bs4 import BeautifulSoup
import requests
import tarfile
import io
import arxiv


def get_wikipedia_intro(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Wikipedia introductory content is usually inside a <p> tag just after the table of contents
    intro_paragraph = soup.select_one('.mw-parser-output > p:not(.mw-empty-elt)')

    return intro_paragraph.text if intro_paragraph else None

# # Test the function
# url = "https://en.wikipedia.org/wiki/Web_scraping"
# print(get_wikipedia_intro(url))

#
# def extract_introduction_from_latex(url):
#     response = requests.get(url, stream=True)
#     introduction_text = ""
#
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Create a file-like object from the streaming content
#         fileobj = io.BytesIO(response.content)
#
#         # Decompress on-the-fly using tarfile
#         with tarfile.open(fileobj=fileobj) as tar:
#             for member in tar.getmembers():
#                 # Assuming the main latex file has the same name as the arXiv ID (this is a heuristic and may need adjustments)
#                 if member.name.endswith('.tex'):
#                     f = tar.extractfile(member)
#                     if f:
#                         content = f.read().decode('utf-8', errors='ignore')
#                         start_idx = content.find('\section{Introduction}')
#                         end_idx = content.find('\section{', start_idx + 1)
#
#                         if start_idx != -1 and end_idx != -1:
#                             introduction_text = content[start_idx:end_idx]
#                             break
#
#     return introduction_text


'''
The goal of the scraping is to find real life problems in science and engineering 
that are solved via transforming the problem into an solved problem in mathematics or physics,
 kind of like the mathematical modeling of the "new" problem. Therefore, for the search, 
 I will need to organize the scrapped files according to different disciplines. 
 The scraping strategy in general is that, for each paper, I want to fetch the problem in the arxiv papers
  and methods to solve it and store them into separate files with an index that can align them well.
  
  
chat-gpt 4 response:{
I understand your objective better now. You're looking to find interdisciplinary papers where real-world problems might be
 formulated and solved using techniques from diverse fields, whether it be physics, mathematics, chemistry, biology, 
 or various engineering disciplines.

"Interdisciplinary modeling": This might yield results where authors explicitly state their work as being interdisciplinary.
"Real-world problem formulation": This can help identify papers where real-world problems are being mathematically or scientifically formulated.
"Cross-disciplinary solution" OR "Cross-disciplinary approach": To find solutions/approaches that span multiple disciplines.
"Problem translation" AND ("chemistry" OR "biology" OR "physics" OR "engineering"): Looking for papers where problems are "translated" into another domain for a solution.
"Mathematical representation" AND ("real-world" OR "practical"): Papers that focus on representing real-world problems mathematically.
"Physical analogy" OR "Engineering analogy": Some papers might use analogies from physics or engineering to explain or solve problems from other domains.
  }
'''

import arxiv
import fitz  # PyMuPDF
import requests

# Function to extract introduction from a PDF
import arxiv
import fitz  # PyMuPDF
import requests
import re
import os

# Function to extract introduction from a PDF
def extract_introduction_from_pdf(pdf_url):
    introduction_text = ""

    # Download the PDF file
    response = requests.get(pdf_url)

    if response.status_code == 200:
        pdf_data = response.content

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()

            # Search for common variations of introduction headings
            introduction_headings = ["Introduction", "1. Introduction", "I. Introduction"]
            for heading in introduction_headings:
                if heading in page_text:
                    intro_start = page_text.index(heading)
                    introduction_text = page_text[intro_start:]
                    break

            # Check if we found an introduction
            if introduction_text:
                break

        pdf_document.close()

    return introduction_text

# Function to fetch and extract introductions from arXiv papers in PDF format
def fetch_and_extract_introductions(search_query, max_results=10, output_folder="introductions"):
    introductions = {}

    # Query arXiv for papers using the provided search criteria
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # For each result, check if a PDF link is available and extract the introduction
    for result in search.results():
        pdf_link = result.pdf_url

        if pdf_link:
            introduction = extract_introduction_from_pdf(pdf_link)

            # Remove excessive whitespace and newlines
            introduction = re.sub(r'\s+', ' ', introduction).strip()

            # Write the introduction to a file named after the article title
            title = result.title.replace(" ", "_")  # Replace spaces with underscores in the title
            filename = os.path.join(output_folder, f"{title}.txt")

            with open(filename, "w", encoding="utf-8") as file:
                file.write(introduction)

            introductions[result.title] = filename

    return introductions

# Test
# search_criteria = "mathematical modeling in physics"
# introductions = fetch_and_extract_introductions(search_criteria, max_results=1000)
#
# # Print the file paths of the saved introductions
# for title, file_path in introductions.items():
#     print(f"Title: {title}\nIntroduction saved to: {file_path}\n")


# # Function to extract introduction from LaTeX source
# def extract_introduction_from_latex(url):
#     introduction_text = ""
#
#     # Download the LaTeX source tar.gz file
#     response = requests.get(url, stream=True)
#
#     if response.status_code == 200:
#         # Create a file-like object from the streaming content
#         fileobj = io.BytesIO(response.content)
#
#         # Decompress on-the-fly using tarfile
#         with tarfile.open(fileobj=fileobj) as tar:
#             for member in tar.getmembers():
#                 # Assuming the main LaTeX file has the name structure of the arXiv ID
#                 if member.name.endswith('.tex'):
#                     f = tar.extractfile(member)
#                     if f:
#                         content = f.read().decode('utf-8', errors='ignore')
#                         start_idx = content.find('\\section{Introduction}')
#                         end_idx = content.find('\\section{', start_idx + 1)
#
#                         if start_idx != -1 and end_idx != -1:
#                             introduction_text = content[start_idx:end_idx]
#                             break
#
#     return introduction_text
#
# # Function to fetch and extract introductions from arXiv papers
# def fetch_and_extract_introductions(search_query, max_results=10):
#     introductions = {}
#
#     # Query arXiv for papers using the provided search criteria
#     search = arxiv.Search(
#         query=search_query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.Relevance
#     )
#
#     # For each result, visit the arXiv preprint page and extract the introduction
#     for result in search.results():
#         arxiv_url = result.entry_id
#         response = requests.get(arxiv_url)
#
#         if response.status_code == 200:
#             breakpoint()
#             soup = BeautifulSoup(response.text, 'html.parser')
#             latex_link = soup.find('a', text='Download source')
#
#             if latex_link:
#                 latex_url = latex_link['href']
#                 introduction = extract_introduction_from_latex(latex_url)
#                 introductions[result.title] = introduction
#
#     return introductions
#
# # Test
# search_criteria = "mathematical modeling in physics"
# introductions = fetch_and_extract_introductions(search_criteria, max_results=5)
# for title, intro in introductions.items():
#     print(f"Title: {title}\nIntroduction: {intro[:500]}...\n\n")  # Printing first 500 chars of intro for brevity
#
#



# import requests
# from bs4 import BeautifulSoup
# import os
# from io import BytesIO
# from pdfminer.high_level import extract_text
#
# URL = "https://www.contest.comap.com/undergraduate/contests/matrix/index.html"
# BASE_URL = "https://www.contest.comap.com/undergraduate/contests/matrix/"
# response = requests.get(URL)
#
# soup = BeautifulSoup(response.content, 'html.parser')

# Find all the links in the webpage
# links = soup.find_all('a')
# breakpoint()
# # Filter out links that point to MCM problems in the "PDF" directory
# mcm_links = [link for link in links if link.get('href', '').startswith("PDF/") and link.get('href', '').endswith(".pdf")]
#
#
# # Create a directory to store the extracted texts
# if not os.path.exists('MCM_Problems'):
#     os.makedirs('MCM_Problems')
#
# # Iterate over each MCM problem link, extract the text from the PDF, and store it in a file
# for link in mcm_links:
#     problem_url = f"{BASE_URL}{link['href']}"
#     problem_response = requests.get(problem_url)
#
#     # Extract text from the PDF
#     pdf_content = BytesIO(problem_response.content)
#     problem_text = extract_text(pdf_content)
#
#     # Extract the problem year and name for the filename
#     filename = link['href'].split('/')[-1].replace('.pdf', '.txt')
#
#     with open(f"MCM_Problems/{filename}", 'w', encoding='utf-8') as file:
#         file.write(problem_text)
#
# print("Extraction completed. Check the 'MCM_Problems' directory for the extracted texts.")

# a=1
# b=2





import requests
from pdfminer.high_level import extract_text
import PyPDF2
from io import BytesIO
import os
from bs4 import BeautifulSoup






URL = "https://pubs.aip.org/aip/jurp/article/32/1/100002/2908481/A-Simple-Model-for-Understanding-Cloud-Diffusion?searchresult=1"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
# Find the desired tag
pdf_link_tag = soup.find("a", class_="article-pdfLink")
# Extract the href attribute (link to the PDF)
relative_pdf_link = pdf_link_tag["href"]
# Construct the complete URL
base_url = "https://pubs.aip.org"
complete_pdf_url = base_url + relative_pdf_link
breakpoint()
pdf_response  = requests.get(URL, headers=headers,allow_redirects=True)
# Create a session
session = requests.Session()
pdf_response = session.get(complete_pdf_url,headers=headers,allow_redirects=True)
if pdf_response.status_code == 200:
    pdf_file_content = BytesIO(pdf_response.content)

    # Use PyPDF2 to read the content
    pdf_reader = PyPDF2.PdfReader(pdf_file_content)

    # Extract text from each page using the new method
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    print(pdf_text)
    with open("sample.txt", "w", encoding="utf-8") as file:
        file.write(pdf_text)
else:
    print(f"Failed to retrieve the PDF. Status code: {pdf_response.status_code}")
# print(pdf_text)
# problem_response = requests.get(URL)
# pdf_content = BytesIO(problem_response.content)
# problem_text = extract_text(pdf_content)
# # Save to a temporary file
# print(problem_text)



# from selenium import webdriver
#
# # Set up the driver (assuming you have chromedriver installed)
# driver = webdriver.Chrome()
#
# # Navigate to the page
# driver.get('YOUR_INITIAL_PAGE_URL')  # Replace with the URL where the PDF link is present
#
# # Find the PDF link element and click it (this is a hypothetical example; you'll need to adjust based on the actual structure)
# pdf_link = driver.find_element_by_css_selector('a.pdf-link')  # Adjust the selector
# pdf_link.click()
#
# # TODO: Handle the PDF download. Selenium itself can't read PDFs, but it can facilitate the download.
#
# driver.quit()
