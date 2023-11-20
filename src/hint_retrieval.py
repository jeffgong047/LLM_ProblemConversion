import chromadb
from langchain.embeddings import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from IPython.testing import test
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
import os

os.chdir('/common/home/hg343/Research/LLM_ProblemConversion/datasets')

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
counter = 0

for filename in os.listdir("Wikipedia"):
    #t = "vieta’s formulas, pythagorean theorem, chinese remainder theorem, angle bisector theorem, binomial theorem, bezout’s theorem, ceva’s theorem, de moivre’s theorem, descartes’ theorem, dirichlet’s theorem on arithmetic progressions, viviani's theorem, wilson's theorem, pick's theorem, pythagorean theorem, intermediate value theorem, bayes' theorem, morley's trisector theorem, fermat's little theorem, descartes's theorem, exterior angle theorem, binomial theorem, brianchon's theorem, rational root theorem, rédei's theorem, miquel's theorem, de gua's theorem, menelaus's theorem, squeeze theorem, dirichlet's approximation theorem, prime number theorem, niven's theorem, thales's theorem, pitot theorem"
    #if(filename.lower() in t):
    file_path = os.path.join("/content/drive/My Drive/Research/Meeting Prep/6Dataset 1/Dataset 2 Refined/Wikipedia/" + filename, filename+"_part1.txt")
    with open(file_path, 'r') as file:
        text = file.read()
        testTextAll.append(text)
        ids.append(filename)
        counter = counter + 1
collection.add(
    documents=testTextAll,
    ids=ids
)

query_texts = []

dataset1Path = "/content/drive/My Drive/Research/Meeting Prep/6Dataset 1/Dataset 1 Small"
solution_path = "/content/drive/My Drive/Research/Meeting Prep/6Dataset 1/SolutionsChromaDBRefined"

numFile = 1
for filename in os.listdir(dataset1Path):
    file_path = os.path.join(dataset1Path, filename)
    with open(file_path, 'r') as file:
        text = file.read()
        query_texts.append(text)

    results = collection.query(
        query_texts=query_texts,
        n_results=3
    )
    response = results['ids'][numFile-1][0] + "\n" + results['ids'][numFile-1][1] + "\n" + results['ids'][numFile-1][2]

    file_name = filename

    file_path = f"{solution_path}/{file_name}"

    if not os.path.exists(solution_path):
        try:
            os.makedirs(solution_path)
        except OSError:
            print(f"Error creating directory: {solution_path}")

    try:
        with open(file_path, 'w') as file:
            file.write(response)
            print("Wrote to: " + file_path)
    except IOError:
        print(f"Error writing to '{file_path}'.")

    print("Problem " + filename)
    print("Top Candidate: " + results['ids'][numFile-1][0] + " " + str(results['distances'][numFile-1][0]))
    #print(results['documents'][printResultNum-1][0] + "\n")
    print("Second Candidate: " + results['ids'][numFile-1][1] + " " + str(results['distances'][numFile-1][1]))
    #print(results['documents'][printResultNum-1][1] + "\n")
    print("Third Candidate: " + results['ids'][numFile-1][2] + " " + str(results['distances'][numFile-1][2]))
    #print(results['documents'][printResultNum-1][2])
    print("\n")

    numFile += 1