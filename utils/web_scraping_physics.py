import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import os
import logging

# Logging setup
logging.basicConfig(filename="scraping_errors.log", level=logging.ERROR)

base_url = "https://pubs.aip.org"
start_url = "https://pubs.aip.org/aapt/ajp/issue"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Directory for saving the articles
root_dir = 'ajp_articles'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

def fetch_response(url, max_retries=3):
    retries = 0
    while retries <= max_retries:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        retries += 1
        logging.error(f"Failed to fetch {url}. Attempt {retries}. Status Code: {response.status_code}")
    return None

response = fetch_response(start_url)
if response:
    soup = BeautifulSoup(response.text, 'html.parser')
    decade_links = [base_url + option['value'] for option in soup.select('#DecadeList option') if 1990 <= int(option.text) <= 2020]

    for decade_link in decade_links:
        decade_response = fetch_response(decade_link)
        if decade_response:
            decade_soup = BeautifulSoup(decade_response.text, 'html.parser')
            year_links = [base_url + option['value'] for option in decade_soup.select('#YearsList option')]

            for year_link in year_links:
                year_response = fetch_response(year_link)
                if year_response:
                    year_soup = BeautifulSoup(year_response.text, 'html.parser')
                    issue_links = [base_url + option['value'] for option in year_soup.select('#IssuesList option')]

                    for issue_link in issue_links:
                        issue_response = fetch_response(issue_link)
                        if issue_response:
                            pdf_soup = BeautifulSoup(issue_response.content, 'html.parser')
                            doi_links = [a['href'] for a in pdf_soup.find_all('a', href=True) if "https://doi.org/" in a['href']]

                            for doi_link in doi_links:
                                doi_response = fetch_response(doi_link)
                                if doi_response:
                                    soup = BeautifulSoup(doi_response.content, 'html.parser')
                                    title_tag = soup.find('title')
                                    if title_tag:
                                        article_name = title_tag.text.split('|')[0].strip()
                                        filename = "".join([c if c.isalnum() else "_" for c in article_name]) + ".txt"
                                    else:
                                        article_name = "Unknown_Title"
                                        filename = "Unknown_Title.txt"

                                    meta_tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
                                    if meta_tag:
                                        pdf_url = meta_tag['content']
                                    else:
                                        pdf_url = None

                                    tries = 3
                                    attempt = 0
                                    pdf_written = False
                                    while not pdf_written and attempt < tries:
                                        try:
                                            pdf_response = fetch_response(pdf_url)
                                            if pdf_response:
                                                pdf_file_content = BytesIO(pdf_response.content)
                                                pdf_reader = PyPDF2.PdfReader(pdf_file_content)
                                                pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])

                                                with open(os.path.join(root_dir, filename), "w", encoding="utf-8") as file:
                                                    file.write(pdf_text)
                                                pdf_written = True
                                            else:
                                                logging.error(f"Failed to retrieve the PDF for {doi_link}. Status code: {pdf_response.status_code}")
                                                attempt += 1
                                        except Exception as e:
                                            attempt += 1
                                            logging.error(f"Error processing PDF for {doi_link}. Error: {e}")


#
#
# import requests
# from bs4 import BeautifulSoup
# import PyPDF2
# from io import BytesIO
# import os
# import json
# import logging
#
# logging.basicConfig(filename='ajp_articles.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
#
# base_url = "https://pubs.aip.org"
# start_url = "https://pubs.aip.org/aapt/ajp/issue"
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# }
#
# # Directory for saving the articles
# root_dir = 'ajp_articles'
# if not os.path.exists(root_dir):
#     os.makedirs(root_dir)
#
# response = requests.get(start_url, headers=headers)
# if response.status_code == 200:
#     soup = BeautifulSoup(response.text, 'html.parser')
#     decade_links = [base_url + option['value'] for option in soup.select('#DecadeList option') if 1990 <= int(option.text) <= 2020]
#
#     for decade_link in decade_links:
#         decade_response = requests.get(decade_link, headers=headers)
#         if decade_response.status_code == 200:
#             decade_soup = BeautifulSoup(decade_response.text, 'html.parser')
#             year_links = [base_url + option['value'] for option in decade_soup.select('#YearsList option')]
#
#             for year_link in year_links:
#                 year_response = requests.get(year_link, headers=headers)
#                 year_soup = BeautifulSoup(year_response.text, 'html.parser')
#                 issue_links = [base_url + option['value'] for option in year_soup.select('#IssuesList option')]
#
#                 for issue_link in issue_links:
#                     response = requests.get(issue_link, headers=headers)
#                     pdf_soup = BeautifulSoup(response.content, 'html.parser')
#                     doi_links = [a['href'] for a in pdf_soup.find_all('a', href=True) if "https://doi.org/" in a['href']]
#
#                     for doi_link in doi_links:
#                         response = requests.get(doi_link, headers=headers)
#                         if response.status_code == 200:
#                             soup = BeautifulSoup(response.content, 'html.parser')
#                             title_tag = soup.find('title')
#                             if title_tag:
#                                 article_name = title_tag.text
#                             else:
#                                 article_name = "Unknown_Title"
#                             try:
#                                 article_name = title_tag.text.split('|')[0].strip()
#                                 filename = "".join([c if c.isalnum() else "_" for c in article_name]) + ".txt"
#                                 meta_tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
#                                 if meta_tag:
#                                     pdf_url = meta_tag['content']
#                                 else:
#                                     pdf_url = None
#                             except Exception as e:
#                                 breakpoint()
#                             session = requests.Session()
#                             pdf_response = session.get(pdf_url, headers=headers, allow_redirects=True)
#                             try:
#                                 if pdf_response.status_code == 200:
#
#                                     pdf_file_content = BytesIO(pdf_response.content)
#                                     pdf_reader = PyPDF2.PdfReader(pdf_file_content)
#                                     pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
#                             except Exception as e:
#                                 breakpoint()
#
#                                 with open(os.path.join(root_dir, filename), "w", encoding="utf-8") as file:
#                                     file.write(pdf_text)
#                             else:
#                                 print(f"Failed to retrieve the PDF. Status code: {pdf_response.status_code}")
