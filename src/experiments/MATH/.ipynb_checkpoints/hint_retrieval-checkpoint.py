import json
import time
from tqdm import tqdm
import argparse
import os
import re
import ast
import numpy
import openai

import os 
print(os.getcwd())
import re
from transformers import AutoTokenizer, AutoModelForCausalLM , LlamaForCausalLM
import time
import guidance
from guidance import models, gen, select

# openai.proxy = "http://..."
# os.environ["OPENAI_API_KEY"] = 'sk-...'


llama2 = models.Transformers("meta-llama/Llama-2-13b-chat-hf")


TRY_CNT = 1

# You can now use these variables in your code

# Define the arguments here
temperature = 0.0
majoritycnt = 1
shots = 8
hintcnt = 2
questioncnt = 8
questiontrycnt = 4
answertrycnt = 4
verbose = True
model = 'llama2-13b'
withcode = False
dataset = 'data/test.jsonl'
problem_level_lower_bound = 1
problem_level_upper_bound = 5
problem_interval_begin = 0
problem_interval_end = 500
inverse_problem_order = True

# You can now use these variables in your code


llm = llama2
#gpt4 = guidance.llms.OpenAI("gpt-4")
#guidance.llm = guidance.llms.OpenAI(model, caching=True)

def simulated_annealing(initial_solution, cost_function, temperature, cooling_rate, stop_temperature):
    current_solution = initial_solution
    current_cost = cost_function(current_solution)
    
    while temperature > stop_temperature:
        new_solution = generate_new_solution(current_solution)
        new_cost = cost_function(new_solution)
        
        cost_diff = new_cost - current_cost
        
        if cost_diff < 0 or random.uniform(0, 1) < exp(-cost_diff / temperature):
            current_solution = new_solution
            current_cost = new_cost
        
        temperature *= cooling_rate

    return current_solution



def try_wrapper(func):
    def inner(*args, **kwargs):
        try_cnt = 0
        while try_cnt < TRY_CNT:
            print('try_cnt',  try_cnt)
            try:
                print('i got here!')
                print('args:', args)
                print('kwargs: ', kwargs)
                print('func', func)
                return func(*args, **kwargs)
            except Exception as e:
                print(f"func() failed, try again... (No. {try_cnt + 1}). Error: {e}")
                try_cnt += 1
                time.sleep(min(1024, 2 ** (try_cnt / 2)))
                continue
    print('return inner')
    return inner


def get_time_str(trycnt=0):
    return "2023-06-01-12-00-" + str(trycnt).zfill(2)
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


examples = []

""
# we can pre-define valid option sets
valid_correctness = ["Correct", "Wrong", "Unknown"]

# Define the guidance program judger, define the {{final_answer}} are correct,
#   given the ground truth {{ground_truth_answer}}


#how to write judger such that it can handle different problems without refreshing the context it has 
@guidance
def judger(llm,*args, **kwargs):
    llm =  llama2 + f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
    Your job is to judge whether the "final answer" is correct based on "ground truth answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. 
    Problem Subject: {kwargs['question_subject']}, Problem Content: {kwargs['question_content']}, Final answer is: {kwargs['final_answer']},Ground truth answer: {kwargs['ground_truth_answer']}. 
    Is the final_answer correct, given the ground truth answer?''' + select(['Correct', 'Wrong', 'Unknown'],name ='correctness')+ f'''Please provide the reason for your judgement and how confident you are about your judgement on the scale from 1-10.'''+gen(name='notes')

    return llm



@guidance
def program(llm, *args, **kwargs):
    llm = llm +  f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.
Here is the prblem, Q:{kwargs['question']} ,H:Please provide few but less than 5 math theorems that are most representative as hints first before generating the solution, please dont provide explanation:''' +gen(max_token=50,stop = '5', name='hints')+f'''S: Please provide the solution based on the hints:''' + gen(max_token=100, name = 'final_solution')
    return llm

def main():
    # Load the data from the JSONL file
    data = []
    with open(dataset, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            if (json.loads(line)['level'] < problem_level_lower_bound): continue
            if (json.loads(line)['level'] > problem_level_upper_bound): continue
            data.append(json.loads(line))
            cnt += 1
            # if (cnt == args.problem_numbers):
            #     break
    data = data[problem_interval_begin:problem_interval_end + 1]
    print(len(data))
    if inverse_problem_order:
        data = data[::-1]

    t = time.localtime()

    complex_prompts = f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
    YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.
    '''
    complex_examples = []
    with open('complex-cot-math.txt', 'r', encoding='utf-8') as f:
        t = f.read().split("\n\n")
        for i in t:
            question = i.split("\nA:")[0].split('Question: ')[-1]
            # print(question)
            solution = "\nA:".join(i.split("\nA: ")[1:]).split("\nThe answer is ")[0]
            # print(answer)
            final_answer = i.split("\nThe answer is ")[-1]
            print(final_answer)
            complex_examples.append({'question': question, 'solution': solution, 'final_answer': final_answer})

    complex_examples = complex_examples[:shots]

#     # Define the guidance program generate hints
#     program = guidance(complex_prompts, examples=complex_examples)
    t = time.localtime()
    # extract 'test' from args.dataset in format 'data/test.jsonl'
    dataset_name = dataset.split('/')[1].split('.')[0]
    # change huggyllama/llama-13b to huggyllama-llama-13b
    model_name = model.replace('/', '-')
    logfilename = 'results/results-math-complex-cot-openai--' + model_name + '--' + dataset_name + '--k_' + str(
        majoritycnt) + '--' + time.strftime("%Y-%m-%d-%H-%M-%S", t) + '.jsonl'
    with open(logfilename, 'w') as f:
        f.write("Model: " + model + "\n")
        f.write("Temperature: " + str(temperature) + "\n")
        f.write("Majority Cnt: " + str(majoritycnt) + "\n")
        f.write("Hint Cnt: " + str(hintcnt) + "\n")
        f.write("Question Cnt: " + str(questioncnt) + "\n")
        f.write("Dataset: MATH - " + dataset + "\n")
        f.write(
            f"Problem Level Interval: [{str(problem_level_lower_bound)}, {str(problem_level_upper_bound)}]\n")
        # f.write(f"Problem Numbers: First {str(args.problem_numbers)} Problems\n")
        f.write(f"Problem Interval: [{str(problem_interval_begin)}, {str(problem_interval_end)}]\n")
        f.write(f"Inverse Problem Order: {str(inverse_problem_order)}\n")
        f.write("--------------------------------\n")
    # Initialize counter for correct answers
    correct_answers = 0
    cnt = 0
    total_cnt = len(data)

    # Iterate over the data from the JSON file and call the solve function
    for example in tqdm(data, desc="Evaluating", unit="example"):
        cnt += 1

        print("-------------------------\n### Example ID: ", example["unique_id"], "\t ( ", cnt, "/", total_cnt, " )")
        print("Problem Level: ", example["level"])
        print("[Problem Subject]: ", example["subject"])
        print("[Problem Content]: ", example["problem"])
        # new Q for every example
        print('example', example)
        try_cnt = 0
        while True:
            try_cnt += 1
            try:
                out = llama2+ try_wrapper(program)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
#                 print('hints: ',out['hints'])
#                 print('solutions: ', out['solution'])
                print('program executed')
                judgement =llama2+ try_wrapper(judger)(question_content=example['problem'],
                                                question_subject=example['subject'], final_answer=out['solution'],
                                                ground_truth_answer=example['answer'])
#                 print("[Final Solution]: ", out['hints'])
#                 print("[Final Answer]: ", out['solution'])
#                 # print("[Ground Truth Solution]: ", example["solution"])
#                 print("[Ground Truth Answer]: ", example["answer"])
#                 print("[Correctness]: ", judgement["correctness"])
                print('we are good')
                break
            except Exception as e:
                print(e)
                time.sleep(min(1024, 2 ** (try_cnt / 2)))
                continue

        correct_answers += (judgement['correctness'] == 'Correct')
        # Calculate and print the running accuracy
        accuracy = correct_answers / cnt

        print("[Running Average Accuracy]: ", accuracy)

        result = {
            "accuracy": accuracy,
            "example_id": example["unique_id"],
            "level": example["level"],
            "problem_subject": example["subject"],
            "problem_content": example["problem"],
            "correctness": judgement["correctness"],
            "generated_hints": out['hints'],
            "final_solution": out['final_solution'],
            "ground_truth_solution": example["solution"],
            "ground_truth_answer": example["answer"],
        }
        breakpoint()
        # Write the result to a JSON file, note that we open the file in append mode ('a')
        with open(logfilename, 'a') as f:
            f.write(json.dumps(result) + '\n')  # write each result as a new line


if __name__ == "__main__":
    main()