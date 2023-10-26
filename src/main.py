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


def load_text(data_path):
    testTextAll = []
    ids = []
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                testTextAll.append(text)
                ids.append(str(counterS))
                ids_to_ProblemName.append(filename)
                assert ids_to_ProblemName[counterS] == filename
    return ids, testTextAll

def sanity_test_shiftLeft(data_path):
    ids = []
    testTextAll = []
    file_list = os.listdir(data_path)
    for idx, filename in enumerate(file_list):
        if filename.endswith('.txt') and file_list[idx-1].endswith('.txt'):
            file_path = os.path.join(data_path, filename)
            file_path_left = os.path.join(data_path,file_list[idx-1])
            with open(file_path_left, 'r') as file:
                text = file.read()
                testTextAll.append(text)
                ids.append(str(idx))
                ids_to_ProblemName.append(filename)
                assert ids_to_ProblemName[idx] == filename
    return ids, testTextAll


def sanity_test_ReplaceProblemContent(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(solution_path, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                if filename == 'intro_Energy_mass_equivalence_from_Maxwell_equations.txt':
                    # put a different documents under the name  intro_The_mass_spectrum_of_quarkonium_using_matrix_mechanics.txt to hack the database
                    # we expect the model to act poor for this datapoint
                    testTextAll.append('')  #The paper proposes to use the matrix mechanics method to solve the Schrödinger equation for the quarkonium system
                #              testTextAll.append(text)
                else:
                    testTextAll.append(text)
                ids.append(str(counterS))
                ids_to_ProblemName.append(filename)
                assert ids_to_ProblemName[counterS] == filename
                counterS = counterS + 1
    return ids, text

def evaluation(query_idx,ids_to_ProblemName,results_ids):
    correct_num = 0
    for index, p in enumerate(query_idx):
        if any(p.lower() ==  ids_to_ProblemName[int(s)].lower() for s in results_ids[index]):
            # print(f'problem description: ', p)
            # print(f'solution retrieved: ', results_ids[index])
            correct_num+=1
    return correct_num/len(ids_to_ProblemName)
            # print('The top three hit ratio is: ',correct_num/len(ids_to_ProblemName))

def robustness_to_Noise(noise_path):
    ids , noise_text = load_text(noise_path)
    collection.add(documents = noise_text,
                   ids = ids
                   )


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

collection = chroma_client.create_collection(name="physics_literature")

ids_to_ProblemName=[]

testTextAll = []
ids = []
counterS = 0
problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/ajp_articles_manual"
solution_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ajp_articles_manual"
solution_path_noise = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia_Math"



#load chroma database with solutions for physics papers
for filename in os.listdir(solution_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(solution_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            testTextAll.append(text)
            ids.append(str(counterS))
            ids_to_ProblemName.append(filename)
            assert ids_to_ProblemName[counterS] == filename
            counterS = counterS + 1

for filename in os.listdir(solution_path_noise):
    if filename.endswith('.txt'):
        file_path = os.path.join(solution_path_noise, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            testTextAll.append(text)
            ids.append(str(counterS))
            ids_to_ProblemName.append(filename)
            assert ids_to_ProblemName[counterS] == filename
            counterS = counterS + 1

collection.add(
    documents=testTextAll,
    ids=ids
)

#store more texts


#print('total number of solutions: ',counterS)
#Test
problemTextAll = []
idp = []

for filename in os.listdir(problem_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(solution_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
            problemTextAll.append(text)
            idp.append(filename)

results = collection.query(
    query_texts=problemTextAll,
    n_results=1
)

# idp_sl , prblemTextAll_sl  = sanity_test_shiftLeft(problem_path)
# idp_rp,  problemTextAll_rp = sanity_test_ReplaceProblemContent(problem_path)



# results_sl = collection.query(
#     query_texts=problemTextAll_sl,
#     n_results=1
# )
# results_sl_ids = results_sl['ids']
#
# hit_ratio_1_sl = evaluation(idp_sl,ids_to_ProblemName,resuls_sl_ids)
# print('The top one hit ratio for shift left is: ', hit_ratio_1_sl)
# results_rp = collection.query(
#     query_texts=problemTextAll_rp,
#     n_results=1
# )
#
# results_rp_ids = results_rp['ids']
#
# hit_ratio_1_rp =evaluation(idp_rp,ids_to_ProblemName,resuls_rp_ids)
# print('The top one hit ratio for replace problem is: ', hit_ratio_1_rp)
#
# breakpoint()

#print('total number of problems: ',counterP)
#Evaluate with hit-ratio top 1

results_ids = results['ids']
results_distance = results['distances']
results_documents = results['documents']  # this variable from results stores the query to the database
correct_num = 0
for index, p in enumerate(idp):
    if any(p.lower() ==  ids_to_ProblemName[int(s)].lower() for s in results_ids[index]):
        print(f'problem description: ', p)
        print(f'solution retrieved: ', results_ids[index])
        correct_num+=1
print('total number of solutions in the database',len(ids_to_ProblemName))
print(correct_num/len(idp))
#assert counterP == counterS










