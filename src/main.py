from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from chromadb.utils import embedding_functions
import chromadb
import os
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain


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

def evaluation(query_idx,ids_to_ProblemName,results_ids):
    correct_num = 0
    for index, p in enumerate(query_idx):
        if any(p.lower() ==  ids_to_ProblemName[int(s)].lower() for s in results_ids[index]):
            # print(f'problem description: ', p)
            # print(f'solution retrieved: ', results_ids[index])
            correct_num+=1
    return correct_num/len(ids_to_ProblemName)
    # print('The top three hit ratio is: ',correct_num/len(ids_to_ProblemName))

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


def robustness_to_Noise(noise_path):
    ids , noise_text = load_text(noise_path)
    collection.add(documents = noise_text,
                   ids = ids
                   )

def few_shot_analogicalPrompt_chromaDB(chromaDB,problem_space_path,ground_truth_path, solutions_space_path,exemplar_prompt):
    ids_to_ProblemName=[]
    testTextAll = []
    ids = []
    counterS = 0
    problem_path = problem_space_path
    solution_path = ground_truth_path
    solution_path_noise = solutions_space_path


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

    chromaDB.add(
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
            file_path = os.path.join(problem_path, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                problemTextAll.append(exemplar_prompt+text)
                idp.append(filename)

    results = chromaDB.query(
        query_texts=problemTextAll,
        n_results=10
    )

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
    print('number of problems answered correctly',correct_num, 'total number of problems',len(idp),'accuracy' ,correct_num/len(idp))
    breakpoint()
    return correct_num/len(idp)


def few_shot_analogicalPrompt_GPT(problem_path,ground_truth_path, solutions_space_path):
    ids_to_ProblemName=[]
    testTextAll = []
    ids = []
    counterS = 0
    problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/Math_manual"
    solution_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ground_truth_Math_manual"
    solution_path_noise = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia_Math"
    os.environ["OPENAI_API_KEY"] = "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" #(My GPT3 key)
    embeddings = OpenAIEmbeddings()
    directory_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia"
    subdirectories = [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]
    subdirectory_names = ", ".join(subdirectories)
    print(subdirectory_names)


    ground_truths = [file for file in os.listdir(ground_truth_path) if file.endswith(".txt")]

    unique_contents = set()

    combined_content = ""

    def strip_and_normalize(content):
        return re.sub(r'\s+', ' ', content).strip().lower()

    for txt_file in ground_truths:
        file_path = os.path.join(ground_truth_path, txt_file)
        with open(file_path, 'r') as file:
            file_content = strip_and_normalize(file.read())
            if file_content not in unique_contents:
                combined_content += file_content
                combined_content += ", "
                unique_contents.add(file_content)

    combined_content = combined_content.rstrip(", ")

    print(combined_content)
    numProblems = 50



    txt_files = [file for file in os.listdir(problem_path) if file.endswith(".txt")]

    for txt_file in txt_files:
        file_path = os.path.join(dataset1Path, txt_file)
        try:
            with open(file_path, 'r') as file:
                problem = file.read()
        except IOError:
            print(f"Error reading '{file_path}'.")

        llm = OpenAI(temperature=.7)
        template = """You are an expert at mathematics.
      Your job is to select the three best theorems most applicable for solving the given problem.
      You do not need to solve the problem.
      You must select your answer from this list of theorems:""" + combined_content + "{question}" #subdirectory_names has too many tokens

        prompt_template = PromptTemplate(input_variables=["question"], template=template)
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)
        response = answer_chain.run("""What 3 theorems are best applicable for solving this question:
      {"""
                                    + problem +
                                    """}
                                    """)

        solution_path = "/content/drive/My Drive/Research/Meeting Prep/6Dataset 1/SolutionsGPT3Refined"
        file_path = f"{solution_path}/{txt_file}"

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


def main():
    os.chdir('/common/home/hg343/Research/LLM_ProblemConversion/datasets')
    #Initialize database
    chroma_client = chromadb.Client()
    collection_zero_shot = chroma_client.create_collection(name="mathManual_zeroshot")
    collection_few_shot_AR = chroma_client.create_collection(name = "mathManual_fewshotAR")
    # problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/Math_manual"
    # ground_truth_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ground_truth_Math_manual"
    # solution_path_noise = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia_Math"
    problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/ajp_articles_manual"
    ground_truth_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ajp_articles_manual"
    solution_path_noise = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia_Math"


    AR_exemplar1 = '(Rectangles and boxes). Suppose that you have established that of all rectangles with a fixed perimeter, the square has maximum area. ' \
                   'By analogy, you conjecture that of all boxes with a fixed surface area, the cube has maximum volume.'
    AR_exemplar2 = ' (Morphine and meperidine). In 1934, the pharmacologist Schaumann was testing synthetic compounds for their anti-spasmodic effect. ' \
                   'These drugs had a chemical structure similar to morphine. He observed that one of the compounds—meperidine, ' \
                   'also known as Demerol—had a physical effect on mice that was previously observed only with morphine:' \
                   ' it induced an S-shaped tail curvature. By analogy, he conjectured that the drug might also share morphine’s narcotic effects. ' \
                   'Testing on rats, rabbits, dogs and eventually humans showed that meperidine, like morphine, was an effective pain-killer'
    AR_exemplar3 = '(Priestley on electrostatic force). In 1769, Priestley suggested that the absence of electrical influence inside a hollow charged spherical shell was evidence that charges attract and repel with an inverse square force. ' \
                   'He supported his hypothesis by appealing to the analogous situation of zero gravitational force inside a hollow shell of uniform density.'
    AR_exemplar4 = ' (Duty of reasonable care). In a much-cited case (Donoghue v. Stevenson 1932 AC 562), ' \
                   'the United Kingdom House of Lords found the manufacturer of a bottle of ginger beer liable for damages to a consumer who became ill as a result of a dead snail in the bottle. ' \
                   'The court argued that the manufacturer had a duty to take “reasonable care” in creating a product that could foreseeably result in harm to the consumer in the absence of such care, and where the consumer had no possibility of intermediate examination.' \
                   ' The principle articulated in this famous case was extended, by analogy, to allow recovery for harm against an engineering firm whose negligent repair work caused the collapse of a lift (Haseldine v. CA Daw & Son Ltd. 1941 2 KB 343). ' \
                   'By contrast, the principle was not applicable to a case where a workman was injured by a defective crane, since the workman had opportunity to examine the crane and was even aware of the defects'
    prompts = 'Here is one examples of using analogical reasoning for solving a problem. Example1: '+ AR_exemplar1  \
              +' When solving following problem, you could mimick the analogical reasoning if necessary.'
    few_shot_AP_Chroma_result = few_shot_analogicalPrompt_chromaDB(collection_few_shot_AR,problem_path,ground_truth_path,solution_path_noise,prompts)
    few_shot_analogicalPrompts(collection_few_shot_AR ,problem_path,ground_truth_path, solution_path_noise,prompts)


if __name__ == "__main__":
    main()




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

#assert counterP == counterS










