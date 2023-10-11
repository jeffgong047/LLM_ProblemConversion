from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from chromadb.utils import embedding_functions
import os

#Initialize database
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection(name="test")
except:
    pass

try:
    chroma_client.delete_collection(name="OpenAI")
except:
    pass

collection = chroma_client.create_collection(name="test")


testTextAll = []
ids = []
counterS = 0
solution_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/ajp_articles_introductions_Problem"
problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ajp_articles_introductions_Solution"


#load chroma database with solutions for physics papers
for filename in os.listdir(solution_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(solution_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            testTextAll.append(text)
            ids.append(filename)
            counterS = counterS + 1

collection.add(
    documents=testTextAll,
    ids=ids
)
print('total number of solutions: ',counterS)
breakpoint()
#Test
problemTextAll = []
idp = []
counterP = 0
for filename in os.listdir(problem_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(solution_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            problemTextAll.append(text)
            idp.append(filename)
            counterP +=1


results = collection.query(
    query_texts=problemTextAll,
    n_results=3
)
print('total number of problems: ',counterP)
#Evaluate with hit-ratio top 3
results_ids = results['ids']
results_distance = results['distances']
results_documents = results['documents']
correct_num = 0
for index, p in enumerate(idp):
    if any(p.lower() == s.lower() for s in results_ids[index]):
        correct_num+=1
assert counterP == counterS
print('The top three hit ratio is: ',correct_num/counterP)



