import requests
import json

endpoint_url = "https://query.wikidata.org/sparql"
#The following query is for searching the math
# query = """
# SELECT ?field ?fieldLabel ?subfield ?subfieldLabel WHERE {
#   ?field wdt:P279 wd:Q395.
#   ?subfield wdt:P279* ?field.
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
# }
# ORDER BY ?field ?subfield
# """



# response = requests.get(endpoint_url, params={'query': query, 'format': 'json'})
# data = response.json()
#
# class Node:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name
#         self.children = []
#
#     def add_child(self, child_node):
#         self.children.append(child_node)
#
#     def __repr__(self, level=0):
#         ret = "\t" * level + repr(self.name) + "\n"
#         for child in self.children:
#             ret += child.__repr__(level + 1)
#         return ret
#
# nodes = {}
# root = Node("Q395", "Field of mathematics")
#
# breakpoint()
# for item in data['results']['bindings']:
#     field_id = item['field']['value'].split("/")[-1]
#     field_label = item['fieldLabel']['value']
#
#     subfield_id = item['subfield']['value'].split("/")[-1]
#     subfield_label = item['subfieldLabel']['value']
#
#     if field_id not in nodes:
#         nodes[field_id] = Node(field_id, field_label)
#
#     if subfield_id not in nodes:
#         nodes[subfield_id] = Node(subfield_id, subfield_label)
#
#     nodes[field_id].add_child(nodes[subfield_id])
#
# print(root)


# import requests
#
# def fetch_subclasses(item_id):
#     query = f"""
#     SELECT ?item ?itemLabel ?itemDescription WHERE {{
#       wd:{item_id} wdt:P279* ?item.
#       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
#     }}
#     """
#
#     url = "https://query.wikidata.org/sparql"
#     headers = {
#         "Accept": "application/sparql-results+json"
#     }
#     response = requests.get(url, headers=headers, params={'query': query})
#     data = response.json()
#     return data['results']['bindings']
#
# item_id = 'Q395' # Example ID for Mathematics
# results = fetch_subclasses(item_id)
#
# with open('wikidata_subclasses.txt', 'w', encoding='utf-8') as file:
#     for result in results:
#         item_label = result['itemLabel']['value']
#         item_description = result['itemDescription']['value'] if 'itemDescription' in result else 'No description'
#         file.write(f"{item_label}: {item_description}\n")

# import requests
#
# query = """
# SELECT ?field ?fieldLabel ?subfield ?subfieldLabel ?subfieldDescription WHERE {
#   ?field wdt:P279 wd:Q395.
#   ?subfield wdt:P279* ?field.
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
# }
# ORDER BY ?field ?subfield
# """
#
# url = "https://query.wikidata.org/sparql"
# headers = {
#     "Accept": "application/sparql-results+json"
# }
# response = requests.get(url, headers=headers, params={'query': query})
# data = response.json().get('results', {}).get('bindings', [])
#
# with open('wikidata_mathematics_fields.txt', 'w', encoding='utf-8') as file:
#     for result in data:
#         field_label = result['fieldLabel']['value']
#         subfield_label = result['subfieldLabel']['value']
#         subfield_description = result.get('subfieldDescription', {}).get('value', 'No description')
#         file.write(f"{field_label} -> {subfield_label}: {subfield_description}\n")
#


import requests

# Base URL for the Wikidata query service
WIKIDATA_URL = "https://query.wikidata.org/sparql"

# Headers for the request
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

# File to store the results
OUTPUT_FILE = "solved_problems.txt"


def run_query(sparql_query):
    """
    Run a SPARQL query against Wikidata and return the results.
    """
    response = requests.get(WIKIDATA_URL, headers=HEADERS, params={"query": sparql_query, "format": "json"})
    return response.json()["results"]["bindings"]


# Start by fetching broad mathematical categories like "Mathematical Theorems" or "Mathematical Problems"
categories_query = """
SELECT ?item ?itemLabel WHERE {
  ?item wdt:P31 wd:Q1936384.  # Instance of "mathematical concept"
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
LIMIT 50
"""
# This is for mathematics
# categories = run_query(categories_query)
#
# # Now, for each category, fetch items that fall under it recursively
# for category in categories:
#     category_label = category["itemLabel"]["value"]
#     category_id = category["item"]["value"].split('/')[-1]  # Extract the QID
#
#     items_query = f"""
#     SELECT ?item ?itemLabel ?itemDescription WHERE {{
#       ?item wdt:P279* wd:{category_id}.
#       SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
#     }}
#     LIMIT 1000
#     """
#
#     items = run_query(items_query)
#
#     # Save the data to a text file
#     with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
#         for item in items:
#             title = item["itemLabel"]["value"]
#             description = item.get("itemDescription", {}).get("value", "No description available.")
#             file.write(f"Title: {title}\n")
#             file.write(f"Description: {description}\n\n")
#
# print(f"Data saved to {OUTPUT_FILE}")


import requests
import time

# Define the API endpoint and headers
url = 'https://query.wikidata.org/sparql'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
}

# Query for main fields of physics
query_fields = """
SELECT ?field ?fieldLabel WHERE {
  ?field wdt:P279 wd:Q413.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""

response = requests.get(url, headers=headers, params={'format': 'json', 'query': query_fields})
fields = response.json()['results']['bindings']

# Create or overwrite the text file
with open('physics_data.txt', 'w', encoding='utf-8') as f:
    for field in fields:
        field_label = field['fieldLabel']['value']
        field_id = field['field']['value'].split('/')[-1]
        f.write(f"Field: {field_label}\n")

        # Query for subfields of the current field
        query_subfields = f"""
        SELECT ?subfield ?subfieldLabel WHERE {{
          ?subfield wdt:P279 wd:{field_id}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """

        response = requests.get(url, headers=headers, params={'format': 'json', 'query': query_subfields})
        subfields = response.json()['results']['bindings']

        for subfield in subfields:
            subfield_label = subfield['subfieldLabel']['value']
            subfield_id = subfield['subfield']['value'].split('/')[-1]
            f.write(f"\tSubfield: {subfield_label}\n")

            # Query for articles or solved problems in the current subfield
            query_articles = f"""
            SELECT ?item ?itemLabel ?itemDescription WHERE {{
              ?item wdt:P361 wd:{subfield_id}.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            """

            response = requests.get(url, headers=headers, params={'format': 'json', 'query': query_articles})
            articles = response.json()['results']['bindings']

            for article in articles:
                article_label = article['itemLabel']['value']
                article_description = article.get('itemDescription', {}).get('value', 'No description available')
                f.write(f"\t\tArticle/Problem: {article_label} - {article_description}\n")

            # Sleep to prevent overwhelming the server
            time.sleep(1)

        # Sleep to prevent overwhelming the server
        time.sleep(1)

print("Data collection finished!")
