import os

import openai
from flask import Flask, redirect, render_template, request, url_for
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
import openai

def get_responses(file_content, questions, max_retries=3):
    # Start with a system message to set the context
    messages = [{"role": "system", "content": f"Based on the following paper:\n\n{file_content}"}]

    # Add each question as a user message
    for _, question in questions.items():
        messages.append({"role": "user", "content": question})
    breakpoint()
    # Retry mechanism for the API call
    attempt = 0
    success = False
    while attempt < max_retries and not success:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            success = True
        except Exception as e:
            print(f"Error querying OpenAI API. Attempt {attempt + 1}. Error: {str(e)}")
            attempt += 1

    if not success:
        print("Max retries reached. Appending empty responses.")
        return [""] * len(questions)

    # Extract the model's replies
    replies = [message['content'] for message in response.choices[0]['message'] if message['role'] == 'assistant']

    return replies


# Initialize the OpenAI API with your API key
openai.api_key = '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'

# Example usage
file_path = './datasets/problems/ajp_articles_introductions/intro_Sliding_and_rolling_along_circular_tracks_in_a_vertical_plane.txt'
questions = {
    "Problem":"Please extract and summarize the core problem that this paper endeavors to solve, using information solely from the content provided within the paper?",
    "Solution":"Please extract and summarize the solution provided by this paper, using information solely from the content provided within the paper?",
    "Used Problem Conversion":"Please provide a binary label indicating whether the problem solving in this paper contains the abstract reasoning step where the problem is converted to some existing known or solved problems?",
    "Reasoning type":"If the problem solving used the heuristics obtained from solving another problem, could you summarize briefly how that is done in the paper, otherwise just briefly describe what type of reasoning the paper did such that the problem could be resolved?",
    "Noise data":"Do you think the paper does not actually attempts to solve a problem which should be removed due to off-topic?"
}

#simple testing
breakpoint()
file_content = ""
with open(file_path, 'r') as file:
    file_content = file.read()

responses = get_responses(file_content, questions)
for resp in responses:
    print(resp)
breakpoint()

#Annotate real data
directory_path = ""
data_set_name = ""

# # Create separate directories for each question
# for question in questions:
#     question_dir = os.path.join( directory_path,f"{data_set_name}_{question.replace(' ', '_')}")
#     os.makedirs(question_dir, exist_ok=True)
#
# for filename in os.listdir(directory_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(directory_path, filename)
#         file_content = ""
#         with open(file_path, 'r') as file:
#             file_content = file.read()
#         responses = get_responses(file_content, questions)
#         problem, solution = responses
#
#         # Create directories based on question if they don't exist
#         for question in questions:
#             question_dir = os.path.join( directory_path,f"{data_set_name}_{question.replace(' ', '_')}")
#             os.makedirs(question_dir, exist_ok=True)
#
#         # Store responses in separate files with the format "paper_name_question.txt"
#         for i, question in enumerate(questions):
#             question_filename = f"{filename}_{question.replace(' ', '_')}.txt"
#             question_path = os.path.join(directory_path,f"{data_set_name}_{question.replace(' ', '_')}", question_filename)
#             with open(question_path, 'w') as question_file:
#                 question_file.write(responses[i])


# Initialize the OpenAI API with your API key




# Annotate real data
directory_path = "./datasets/problems"
data_set_name = "ajp_articles_introductions"

# Create separate directories for each question
for question in questions:
    question_dir = os.path.join(directory_path, f"{data_set_name}_{question.replace(' ', '_')}")
    os.makedirs(question_dir, exist_ok=True)

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        file_content = ""
        with open(file_path, 'r') as file:
            file_content = file.read()

        responses = get_responses(file_content, questions)

        # Store responses in separate files with the format "paper_name_question.txt"
        for i, (key, question) in enumerate(questions.items()):
            question_filename = f"{filename}_{question.replace(' ', '_')}.txt"
            question_path = os.path.join(directory_path, f"{data_set_name}_{question.replace(' ', '_')}", question_filename)
            with open(question_path, 'w') as question_file:
                question_file.write(responses[i])
