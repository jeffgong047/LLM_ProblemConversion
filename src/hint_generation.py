from langchain.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
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
import logging
#from LLM_ProblemConversion.utils import load_text , hit_ratio_test





import os
import re
#from openai.embeddings_utils import OpenAIEmbeddings
#from openai.llm import OpenAI, LLMChain, PromptTemplate

def hint_generation(problem_path, ground_truth_path, solution_space_path, model):
    # Set environment variable for API key
    os.environ["OPENAI_API_KEY"] = "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"
    # Collect subdirectories
    subdirectories = [item for item in os.listdir(solution_space_path)
                      if os.path.isdir(os.path.join(solution_space_path, item))]
  #  print(", ".join(subdirectories))

    # Function to strip and normalize content
    def strip_and_normalize(content):
        return re.sub(r'\s+', ' ', content).strip().lower()

    # Process ground truths
    unique_contents = set()
    for txt_file in os.listdir(ground_truth_path):
        if txt_file.endswith(".txt"):
            file_path = os.path.join(ground_truth_path, txt_file)
            with open(file_path, 'r') as file:
                file_content = strip_and_normalize(file.read())
                unique_contents.add(file_content)
    hint_pool = ", ".join(unique_contents)
   # print(hint_pool)

    # Initialize LLM
    llm = OpenAI(temperature=.7)
    template = f"""You are an expert at mathematics. Your job is to select the three best theorems most applicable for solving the given problem.
     You do not need to solve the problem. You must select your answer from this list of theorems: {hint_pool}{{question}}"""

    # Process problems
    solution_path = "./../datasets/inference_result"
    hit_count, total_files = 0, 0
    for problem in os.listdir(problem_path):
        if problem.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(problem_path, problem)
            try:
                with open(file_path, 'r') as file:
                    problem_txt = file.read()
            except IOError:
                print(f"Error reading '{file_path}'.")
                continue
            prompt_template = PromptTemplate(input_variables=["question"], template=template)
            answer_chain = LLMChain(llm=llm, prompt=prompt_template)
            response = answer_chain.run(f"What 3 theorems are best applicable for solving following question? Please provide response that are separated by , only. \n{{\n{problem_txt}\n}}\n")

            # Save response to file
            output_file_path = os.path.join(solution_path, problem)
            os.makedirs(solution_path, exist_ok=True)
            try:
                with open(output_file_path, 'w') as file:
                    file.write(response)
                    print("Wrote to: " + output_file_path)
            except IOError:
                print(f"Error writing to '{output_file_path}'.")

            # Compare with ground truth
            ground_truth_file_path = os.path.join(ground_truth_path, problem)
            if os.path.isfile(ground_truth_file_path):
                with open(ground_truth_file_path, 'r',encoding='utf-8-sig') as ground_truth_file:
                    ground_truth_items = [strip_and_normalize(item) for item in ground_truth_file.read().split(',')]
                    response =[strip_and_normalize(item) for item in response.split(',')]
                for item in ground_truth_items:
                    item = item.strip('\'')
                    print('ground_truth_items: ', ground_truth_items)
                    print('response: ', response)
                    if item in response:
                        hit_count += 1
                        logging.info('response contains ground truth', 'hit_count:', hit_count)
                        break
                    else:
                        logging.info('response does not include ground_truth')

    logging.info('hit count: ',hit_count,'total problems: ',total_files)
    return f"{hit_count}/{total_files}"

def main():
    logging.basicConfig(filename=f'hint_generation_davinci003_test.log', level=logging.DEBUG)
    os.chdir('/common/home/hg343/Research/LLM_ProblemConversion/datasets')
    problem_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/Math_manual"
    ground_truth_path = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/ground_truth_Math_manual"
    solution_path_noise = "/common/home/hg343/Research/LLM_ProblemConversion/datasets/solution/Wikipedia"
  #  model  = get_LLM_model() # we might uses lang-chain to implement this
    hint_accuracy = hint_generation(problem_path,ground_truth_path,solution_path_noise, model=None)
    print(hint_accuracy)
if __name__ == "__main__":
    main()

