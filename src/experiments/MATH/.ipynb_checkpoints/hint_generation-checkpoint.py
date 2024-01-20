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

# openai.proxy = "http://..."
# os.environ["OPENAI_API_KEY"] = 'sk-...'

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
model = 'llama2_13b'
withcode = False
dataset = 'data/test.jsonl'
problem_level_lower_bound = 1
problem_level_upper_bound = 5
problem_interval_begin = 0
problem_interval_end = 500
inverse_problem_order = True



import os
from guidance import user, assistant,system
os.environ["OPENAI_API_KEY"] = "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"

gpt4 = models.OpenAI("gpt-4")


# You can now use these variables in your code

#gpt4 = guidance.llms.OpenAI("gpt-4")
#guidance.llm = guidance.llms.OpenAI(model, caching=True)

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
    with system():
        lm = llm+ f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
    Your job is to judge whether the "final answer" is correct based on "ground truth answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct.'''
    with user():
        lm += f'''Problem Subject: {kwargs['question_subject']}, Problem Content: {kwargs['question_content']}, Final answer is: {kwargs['final_answer']},Ground truth answer: {kwargs['ground_truth_answer']}. 
    Is the final_answer correct, given the ground truth answer? Please select from one of following option[Correct', 'Wrong', 'Unknown]''' 
    with assistant():
        lm += gen(name='Correctness')
    return lm


@guidance
def hint(llm,*args,**kwargs):
    with system():
        lm = llm+f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.'''
    with user():
        lm += f'''Here is the prblem, Q:{kwargs['question']} ,H:Please provide few but less than 5 math theorems that are most representative as hints first before generating the solution, please dont provide explanation.'''
    with assistant():
        lm += gen(max_tokens=200, name='hints')
    return lm 

@guidance
def program(llm,*args, **kwargs):
    lm = llm+  f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.'''+ f'''Here is the prblem, Q:{kwargs['question']} ,and please use these hint to solve the problem: {kwargs['hints']}. S: Please provide the solution based on the hints:'''  + gen(max_tokens=1000, name = 'final_solution')
    return lm
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
    logfilename = 'results/results-math-hint-generation' + model_name + '--' + dataset_name + '--k_' + str(
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
                breakpoint()
                hints = gpt4 + try_wrapper(hint)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
                out = llama2+ try_wrapper(program)(hints=hints['hints'],question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
#                 print('hints: ',out['hints'])
#                 print('solutions: ', out['solution'])
                print('program executed')
                judgement = gpt4+ try_wrapper(judger)(question_content=example['problem'],
                                                question_subject=example['subject'], final_answer=out['final_solution'],
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
        try:
            correct_answers += (judgement['Correctness'].lower() == 'correct')
        except Exception as e:
            pass
        # Calculate and print the running accuracy
        accuracy = correct_answers / cnt

        print("[Running Average Accuracy]: ", accuracy)

        result = {
            "accuracy": accuracy,
            "example_id": example["unique_id"],
            "level": example["level"],
            "problem_subject": example["subject"],
            "problem_content": example["problem"],
            "correctness": judgement["Correctness"],
            "generated_hints": hints['hints'],
            "final_solution": out['final_solution'],
            "ground_truth_solution": example["solution"],
            "ground_truth_answer": example["answer"],
        }
        print(result)

        # Write the result to a JSON file, note that we open the file in append mode ('a')
        with open(logfilename, 'a') as f:
            f.write(json.dumps(result) + '\n')  # write each result as a new line


if __name__ == "__main__":
    main()