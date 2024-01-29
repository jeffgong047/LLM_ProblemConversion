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
import sys
# openai.proxy = "http://..."
# os.environ["OPENAI_API_KEY"] = 'sk-...'


llama2 = models.Transformers("microsoft/phi-2")

# openai.proxy = "http://..."
# os.environ["OPENAI_API_KEY"] = 'sk-...'

TRY_CNT = 3

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
        lm += f'''Problem Subject: {kwargs['question_subject']}, Problem Content: {kwargs['question_content']}.
    Is the final_answer: '{kwargs['final_answer']}' correct, given the ground truth answer: '{kwargs['ground_truth_answer']}'? Please select from one of following option[Correct', 'Wrong', 'Unknown]'''
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
    with user():
        lm +=f'''How confident are you about the hints provided from a scale of 1-5, one to five?'''
    with assistant():
        lm += gen(max_tokens=10 , name='confidence')
    return lm 

@guidance
def guided_student(llm,*args, **kwargs):
    lm = llm+  f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Let's think step by step.''' \
         + f'''Here is the prblem, Q:{kwargs['question']}, please provide the solution based on these hints: {kwargs['hints']}'''+ gen(max_tokens=500, name = 'final_solution')
    return lm
#
# @guidance
# def self_guided_student(llm,*args,**kwargs):
#     lm = llm+  f'''
#     YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled.
#     You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
#    Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.'''+ f'''Here is the problem, Q:{kwargs['question']} ,and Please provide few but less than 5 math theorems that are most representative as hints first before generating the solution.'''+gen(max_tokens=2, name='hints')+ f'''S: Please provide the solution based on the self-generated hints:'''  + gen(max_tokens=10, name = 'final_solution')
#     return lm

@guidance
def vanilla_student(llm, *args, **kwargs):
    # Check if 'question' key exists in kwargs
    if 'question' in kwargs:
        lm = llm + f'YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Lets think step by step. Here is the problem, Q:{kwargs["question"]} and please provide the solution' + gen(max_tokens=500, name='final_solution')
    else:
        lm = llm  # Handle the case when 'question' key is missing

    return lm

@guidance
def teacher(llm,*args,**kwargs):
    with system():
        lm = llm+f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Lets think step by step.'''

    with user():
        lm += f'''Please provide solution to the problem, Q:{kwargs['question']}.'''
    with assistant():
        lm += gen(max_tokens=500, name='final_solution')
    return lm


# @guidance
# def self_guided_teacher(llm,  *args,**kwargs):
#     lm = llm+  f'''
#     YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled.
#     You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
#    Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.'''+ f'''Here is the problem, Q:{kwargs['question']} ,and Please provide few but less than 5 math theorems that are most representative as hints first before generating the solution.'''+gen(max_tokens=200, name='hints')+ f'''S: Please provide the solution based on the self-generated hints:'''  + gen(max_tokens=1000, name = 'final_solution')
#     return lm


def extract_Keyword(input_string,keyword_options):
    content = input_string.split()
    keyword_list = []
    # Check if the substring_to_check is in the list of substrings
    for keyword in keyword_options:
        if keyword in content:
            keyword_list.append(keyword)
    assert len(keyword_list)==1
    return keyword_list[0]


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
    logfilename = 'results/new' + model_name + '--' + dataset_name + '--k_' + str(
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
    correct_answers_vanilla = 0
    correct_answers_self_guided = 0
    correct_answers_guided = 0
    correct_answers_teacher = 0
    cnt = 0
    total_cnt = len(data)
    # Iterate over the data from the JSON file and call the solve function
    problem_missed = []
    counter = 0
    for index, example in tqdm(enumerate(data),desc="Evaluating", unit="example"):
        counter +=1
        if counter >50:
            sys.exit()
        cnt += 1
        if example['unique_id'] != "test/algebra/2470.json":
            continue
        else:
            breakpoint()
        print("-------------------------\n### Example ID: ", example["unique_id"], "\t ( ", cnt, "/", total_cnt, " )")
        print("Problem Level: ", example["level"])
        print("[Problem Subject]: ", example["subject"])
        print("[Problem Content]: ", example["problem"])
        # new Q for every example
        print('example', example)

        try:
            hints = gpt4 + try_wrapper(hint)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
            vanilla_student_answer = llama2+ try_wrapper(vanilla_student)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
 #           self_guided_student_answer= llama2+ try_wrapper(self_guided_student)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
            guided_student_answer= llama2+ try_wrapper(guided_student)(hints=hints['hints'],question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
            teacher_answer = gpt4 + try_wrapper(teacher)(question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)
  #          self_guided_teacher_answer = gpt4 + try_wrapper(self_guided_teacher)(hints=hints['hints'],question=example['problem'], sol_temperature=temperature, ans_temperature=temperature)

#
            #                 print('hints: ',out['hints'])
#                 print('solutions: ', out['solution'])
            print('program executed')
            judgement_vanilla = gpt4+ try_wrapper(judger)(question_content=example['problem'],
                                            question_subject=example['subject'], final_answer=vanilla_student_answer['final_solution'],
                                            ground_truth_answer=example['answer'])
            # judgement_self_guided =gpt4+ try_wrapper(judger)(question_content=example['problem'],
            #                                                  question_subject=example['subject'], final_answer=self_guided_student_answer['final_solution'],
            #                                                  ground_truth_answer=example['answer'])
            judgement_guided  = gpt4+ try_wrapper(judger)(question_content=example['problem'],
                                                          question_subject=example['subject'], final_answer=guided_student_answer['final_solution'],
                                                          ground_truth_answer=example['answer'])
            judgement_teacher =  gpt4+ try_wrapper(judger)(question_content=example['problem'],
                                                           question_subject=example['subject'], final_answer=teacher_answer['final_solution'],
                                                           ground_truth_answer=example['answer'])
            # judgement_self_guided_teacher =  gpt4+ try_wrapper(judger)(question_content=example['problem'],
            #                                                            question_subject=example['subject'], final_answer=self_guided_teacher_answer['final_solution'],
            #                                                            ground_truth_answer=example['answer'])
        except Exception as e:
            problem_missed.append(index)
            print(e)
            time.sleep(min(1024, 2 ** (1 / 2)))
            continue
        try:
            key_word_options = ['Correct', 'Wrong', 'Unknown']
            correct_answers_vanilla += (extract_Keyword(judgement_vanilla['Correctness'],key_word_options).lower() == 'correct')
            # print('vanilla answers the problem: ',extract_Keyword(judgement_vanilla['Correctness']))
            # correct_answers_self_guided +=(judgement_self_guided['Correctness'].lower() =='correct')
            # print('self_guided student answers the problem: ',self_guided_student['Correctness'])
            correct_answers_guided += (extract_Keyword(judgement_guided['Correctness'], key_word_options).lower() =='correct')
            # print('guided student answers the problem: ', extract_Keyword(judgement_guided['Correctness']))
            correct_answers_teacher +=(extract_Keyword(judgement_teacher['Correctness'], key_word_options).lower() =='correct')
            # print('teacher answers the problem: ', extract_Keyword(judgement_teacher['Correctness']))
            # correct_answers_self_guided_teacher +=(judgement_self_guided_teacher['Correctness'].lower() =='correct')
            # print('self guided teacher answers the problem: ', judgement_self_guided_teacher['Correctness'])
        except Exception as e:
            pass
        # Calculate and print the running accuracy
        accuracy_guided = correct_answers_guided / cnt
        # accuracy_self_guided = correct_answers_self_guided/ cnt
        accuracy_vanilla = correct_answers_vanilla / cnt
        accuracy_teacher = correct_answers_teacher/cnt
        # accuarcy_self_guided_teacher = correct_answers_self_guided_teacher/cnt
        try:
            result = {
                "accuracy_guided": accuracy_guided,
                # "accuracy_self_guided":accuracy_self_guided,
                "accuracy_vanilla": accuracy_vanilla,
                "accuracy_teacher": accuracy_teacher,
                # "accuracy_self_guided_teacher":accuarcy_self_guided_teacher,
                "example_id": example["unique_id"],
                "level": example["level"],
                "problem_subject": example["subject"],
                "problem_content": example["problem"],
                "vanilla_correctness": judgement_vanilla["Correctness"],
                "guided_correctness": judgement_guided['Correctness'],
                "teacher_correctness": judgement_teacher['Correctness'],
                "teacher_hints": hints['hints'],
                # "student_hints": self_guided_student_answer['hints'],
                "guided_student_answer": guided_student_answer['final_solution'],
            #    "self_guided_student_answer": self_guided_student_answer['final_solution'],
                "vanilla_student_answer": vanilla_student_answer['final_solution'],
                "teacher solution:":teacher_answer['final_solution'],
                # "judger_conclusion_vanilla":judgement_vanilla['Correctness']  ,
                # "judger_conclusion_guided_student":judgement_guided['Correctness'],
                # "judger_conclusion_teacher": judgement_teacher['Correctness'],

         #       "self_guided teacher solution": self_guided_teacher['final_solution'],
                "confidence:": hints['confidence'],
                "ground_truth_solution": example["solution"],
                "ground_truth_answer": example["answer"],
            }
        except:
            pass

        # Write the result to a JSON file, note that we open the file in append mode ('a')
        with open(logfilename, 'a') as f:
            f.write(json.dumps(result) + '\n')  # write each result as a new line


if __name__ == "__main__":
    main()