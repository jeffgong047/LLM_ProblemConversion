# MATH with Complex CoT

import guidance
from guidance import models, gen, select
import json
import time
from tqdm import tqdm
import argparse
import os
import re
import ast
import sympy
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM , LlamaForCausalLM
import time


# openai.proxy = "http://..."
# os.environ["OPENAI_API_KEY"] = 'sk-...'

TRY_CNT = 16


def get_parser():
    parser = argparse.ArgumentParser(description="Cumulative Reasoning")
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--majoritycnt', type=int, choices=range(1, 101), default=1,
                        help='numbers of majority voting times')
    parser.add_argument('--shots', type=int, choices=range(1, 101), default=8, help='numbers of few-shot examples')
    parser.add_argument('--hintcnt', type=int, choices=range(0, 101), default=2, help='numbers of hints to generate')
    parser.add_argument('--questioncnt', type=int, choices=range(0, 101), default=8,
                        help='numbers of questions to generate')
    parser.add_argument('--questiontrycnt', type=int, choices=range(0, 101), default=4,
                        help='numbers of tries to generate questions')
    parser.add_argument('--answertrycnt', type=int, choices=range(0, 101), default=4, help='numbers of tries to answer')
    parser.add_argument('--verbose', type=ast.literal_eval, default=True, help='verbose mode')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k-0613', help='model to use')
    parser.add_argument('--withcode', type=ast.literal_eval, default='False', help='whether to use code to verify answers')
    parser.add_argument('--dataset', type=str, default='data/test.jsonl', help='dataset to use')
    parser.add_argument('--problem_level_lower_bound', type=int, default=1,
                        help='lower bound of problem level [lower_bound, upper_bound]')
    parser.add_argument('--problem_level_upper_bound', type=int, default=5,
                        help='upper bound of problem level [lower_bound, upper_bound]')
    # parser.add_argument('--problem_numbers', type=int, default=500, help='problem numbers to be evaluated')
    parser.add_argument('--problem_interval_begin', type=int, default=0, help='problem interval begin [begin, end]')
    parser.add_argument('--problem_interval_end', type=int, default=500, help='problem interval end [begin, end]')
    parser.add_argument('--inverse_problem_order', type=ast.literal_eval, default=True, help='whether to inverse problem order')
    return parser


parser = get_parser()
args = parser.parse_args()

# breakpoint()
# # load a model (could be Transformers, LlamaCpp, VertexAI, OpenAI...)
# llama2 = guidance.models.LlamaCpp('/common/home/zw393/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf') 
# # append text or generations to the model
# llama2 + f'Do you want a joke or a poem? ' + gen(stop='.')

llama2 = models.Transformers("meta-llama/Llama-2-13b-chat-hf")



def try_wrapper(func):
    def inner(*args, **kwargs):
        try_cnt = 0
        while try_cnt < TRY_CNT:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"func() failed, try again... (No. {try_cnt + 1}). Error: {e}")
                try_cnt += 1
                time.sleep(min(1024, 2 ** (try_cnt / 2)))
                continue

    return inner


def get_time_str(trycnt=0):
    return "2023-06-01-12-00-" + str(trycnt).zfill(2)
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


examples = [
]

""
# we can pre-define valid option sets
valid_correctness = ["Correct", "Wrong", "Unknown"]

# Define the guidance program judger, define the {{final_answer}} are correct,
#   given the ground truth {{ground_truth_answer}}
llm_judger = llama2+
    """
    {{#system}}YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step. {{/system}}
    {{#system}}Your job is to judge whether the "final_answer" is correct based on "ground_truth_answer", do not be strict on the format, but check the content. Notice that unsolved half results are not Correct. {{/system}}
    {{#system}}Problem Subject: "{{question_subject}}", Problem Content: "{{question_content}}" {{/system}}
    {{#system}}Is the final_answer correct, given the ground truth answer? Reply with Correct, Wrong or Unknown. 
    "final_answer": "{{final_answer}}", "ground_truth_answer": "{{ground_truth_answer}}"{{/system}}
    {{#assistant}}{{select "correctness" options=valid_correctness}}{{/assistant}}
    """


def main():
    # Load the data from the JSONL file
    data = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            if (json.loads(line)['level'] < args.problem_level_lower_bound): continue
            if (json.loads(line)['level'] > args.problem_level_upper_bound): continue
            data.append(json.loads(line))
            cnt += 1
            # if (cnt == args.problem_numbers):
            #     break
    data = data[args.problem_interval_begin:args.problem_interval_end + 1]
    print(len(data))
    if args.inverse_problem_order:
        data = data[::-1]

    t = time.localtime()

    complex_prompts = '''
    {{#system}}
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. You THINK NATURAL, BROAD AND DEEP. Let's think step by step.
    YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational hints that are useful for solving the problem. Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.
    {{/system}}
    {{~#each examples}}
    {{#user}}
    Question:
    {{this.question}}
    A:
    {{/user}}
    {{#assistant}}{{this.solution}}{{/assistant}}
    {{#user}}
    Final Answer:
    {{/user}}
    {{#assistant}}{{this.final_answer}}{{/assistant}}
    {{~/each}}
    
    {{#user}}Question: {{question}}{{/user}}
    {{#assistant}}{{gen "final_solution" temperature=sol_temperature max_tokens=800}}{{/assistant}}
    {{#user}}
    Final Answer:
    {{/user}}
    {{#assistant}}{{gen "final_answer" temperature=ans_temperature max_tokens=50}}{{/assistant}}
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

    complex_examples = complex_examples[:args.shots]

    # Define the guidance program generate hints
    program = llama2+ complex_prompts + complex_examples

    t = time.localtime()

    # extract 'test' from args.dataset in format 'data/test.jsonl'
    dataset_name = args.dataset.split('/')[1].split('.')[0]
    # change huggyllama/llama-13b to huggyllama-llama-13b
    model_name = args.model.replace('/', '-')
    logfilename = 'results/results-math-complex-cot-openai--' + model_name + '--' + dataset_name + '--k_' + str(
        args.majoritycnt) + '--' + time.strftime("%Y-%m-%d-%H-%M-%S", t) + '.jsonl'
    with open(logfilename, 'w') as f:
        f.write("Model: " + args.model + "\n")
        f.write("Temperature: " + str(args.temperature) + "\n")
        f.write("Majority Cnt: " + str(args.majoritycnt) + "\n")
        f.write("Hint Cnt: " + str(args.hintcnt) + "\n")
        f.write("Question Cnt: " + str(args.questioncnt) + "\n")
        f.write("Dataset: MATH - " + args.dataset + "\n")
        f.write(
            f"Problem Level Interval: [{str(args.problem_level_lower_bound)}, {str(args.problem_level_upper_bound)}]\n")
        # f.write(f"Problem Numbers: First {str(args.problem_numbers)} Problems\n")
        f.write(f"Problem Interval: [{str(args.problem_interval_begin)}, {str(args.problem_interval_end)}]\n")
        f.write(f"Inverse Problem Order: {str(args.inverse_problem_order)}\n")
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

        try_cnt = 0
        while True:
            try_cnt += 1
            try:
                breakpoint()
                out = try_wrapper(program)(question=example['problem'], sol_temperature=args.temperature, ans_temperature=args.temperature)

                judgement = try_wrapper(judger)(question_content=example['problem'],
                                                question_subject=example['subject'], final_answer=out['final_answer'],
                                                ground_truth_answer=example['answer'])

                print("[Final Solution]: ", out['final_solution'])
                print("[Final Answer]: ", out['final_answer'])
                # print("[Ground Truth Solution]: ", example["solution"])
                print("[Ground Truth Answer]: ", example["answer"])
                print("[Correctness]: ", judgement["correctness"])
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
            "final_solution": out['final_solution'],
            "final_answer": out['final_answer'],
            "ground_truth_solution": example["solution"],
            "ground_truth_answer": example["answer"],
        }

        # Write the result to a JSON file, note that we open the file in append mode ('a')
        with open(logfilename, 'a') as f:
            f.write(json.dumps(result) + '\n')  # write each result as a new line


if __name__ == "__main__":
    main()

