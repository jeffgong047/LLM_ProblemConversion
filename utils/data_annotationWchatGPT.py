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
#
# def get_responses(file_content, questions, max_retries=3):
#     # Start with a system message to set the context
#     messages = [{"role": "system", "content": f"Based on the following paper:\n\n{file_content}"}]
#
#     # Add each question as a user message
#     for _, question in questions.items():
#         messages.append({"role": "user", "content": question})
#     breakpoint()
#     # Retry mechanism for the API call
#     attempt = 0
#     success = False
#     while attempt < max_retries and not success:
#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages
#             )
#             success = True
#         except Exception as e:
#             print(f"Error querying OpenAI API. Attempt {attempt + 1}. Error: {str(e)}")
#             attempt += 1
#
#     if not success:
#         print("Max retries reached. Appending empty responses.")
#         return [""] * len(questions)
#
#     # Extract the model's replies
#     replies = [message['content'] for message in response.choices[0]['message'] if message['role'] == 'assistant']
#
#     return replies




import openai

import openai

# def get_responses(file_content, questions):
#     # Start with a system message to set the context
#     file_content = file_content[:8500]
#
#     messages = [{"role": "system", "content": f"Based on the following paper, please answer a set of questions under corresponding categories. The feedback should strictly follow the format of 'Category: Answer'. "}]
#     messages.append({"role": "user", "content":f"Here's the paper content: {file_content}"})
#     # Add each question with a unique identifier
#     for identifier, question in questions.items():
#         messages.append({"role": "user", "content": f"({identifier}): {question}"})
#     print('messages: ', messages)
#     # Make the API call
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature  = 0.2,
#             max_tokens = 1500
#
#         )
#         print('response: ',response)
#         # Extract the model's replies based on identifiers
#         output_text = response.choices[0].message.content
#         # Extract the model's replies based on identifiers
#         replies = {}
#         for identifier in questions:
#             # Extract answers by finding the identifier and extracting the content that follows it
#             start_idx = output_text.find(f"Category: {identifier}")
#             if start_idx != -1:
#                 end_idx = output_text.find("Category:", start_idx + 1)  # Find the start of the next category
#                 if end_idx == -1:  # If this is the last category
#                     end_idx = len(output_text)
#                 answer = output_text[start_idx:end_idx].replace(f"Category: {identifier}\nAnswer:", "").strip()
#                 replies[identifier] = answer
#         return replies
#
#     except Exception as e:
#         print(f"Error querying OpenAI API. Error: {str(e)}")
#         return {}

def get_responses(file_content, questions):
    error_counts = 0
    # Trim the file_content to a max length of 8500 characters
    file_content = file_content[:8500]

    # Construct the initial prompt
    prompt_text = f"Based on the following paper, please answer a set of questions under corresponding categories. The feedback should strictly follow the format of 'Category: Answer'. Here's the paper content:\n\n{file_content}\n\n"

    # Add each question with a unique identifier
    for identifier, question in questions.items():
        prompt_text += f"({identifier}): {question}\n"
    print(prompt_text)
    # Make the API call
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt_text,
            temperature=0.2,
            max_tokens=1500
        )

        output_text = response.choices[0].text.strip()
        print(output_text)
        # Extract the model's replies based on identifiers
        replies = {}
        identifiers = list(questions.keys())  # Get a list of all identifiers

        for idx, identifier in enumerate(identifiers):
            start_idx = output_text.find(f"{identifier}:")
            if start_idx != -1:
                # Check if this is the last identifier or get the start index of the next identifier
                end_idx = output_text.find(f"{identifiers[idx+1]}:", start_idx + 1) if idx + 1 < len(identifiers) else len(output_text)
                answer = output_text[start_idx:end_idx].replace(f"{identifier}:", "").strip()
                replies[identifier] = answer

        return replies

        return replies
    except Exception as e:
        error_counts+=1
        print(f"Error querying OpenAI API. Error: {str(e)}")
        get_responses(file_content,questions)
        if error_counts>=3:
            pass

# Initialize the OpenAI API with your API key
openai.api_key = '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'

# Example usage
file_path = './datasets/problems/ajp_articles_introductions/intro_Sliding_and_rolling_along_circular_tracks_in_a_vertical_plane.txt'
questions = {
    "Problem": "From the introduction, please extract verbatim any sections that describe the problem the paper addresses, ensuring you exclude any mention of the solution or the author's approach.",
    "Solution": "How does the paper propose to solve this problem?",
    "Used Problem Conversion": "Did this paper use abstract reasoning to map the problem to a known solution? (Answer with 0 for 'No' and 1 for 'Yes', followed by a brief explanation).",
    "Reasoning type": "If abstract reasoning was used based on another problem's solution, briefly describe how. If not, describe the paper's reasoning approach.",
    "Noise data": "Does this paper solves a problem? (Answer with 0 for 'No' and 1 for 'Yes', followed by a brief explanation if needed)."
}

# file_content = ""
# with open(file_path, 'r') as file:
#     file_content = file.read()
#
# responses = get_responses(file_content, questions)
# print(responses)
#
# breakpoint()





# Annotate real data
directory_path = "./datasets/problems"
data_set_name = "ajp_articles_introductions"

# Create separate directories for each question
for question in questions:
    question_dir = os.path.join(directory_path, f"{data_set_name}_{question.replace(' ', '_')}")
    os.makedirs(question_dir, exist_ok=True)

for filename in os.listdir(os.path.join(directory_path,data_set_name)):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, data_set_name,filename)
        file_content = ""
        with open(file_path, 'r') as file:
            file_content = file.read()

        responses = get_responses(file_content, questions)
        # Store responses in separate files with the format "paper_name_question.txt"
        for i, (question_category, solution) in enumerate(responses.items()):
            question_filename = f"{filename}_{question_category.replace(' ', '_')}.txt"
            question_path = os.path.join(directory_path, f"{data_set_name}_{question_category.replace(' ', '_')}", question_filename)
            with open(question_path, 'w') as question_file:
                question_file.write(solution)
