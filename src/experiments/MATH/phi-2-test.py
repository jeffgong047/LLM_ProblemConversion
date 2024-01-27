import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from guidance import gen
import guidance

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

@guidance
def guided_student(llm,*args, **kwargs):
    breakpoint()
    lm = llm+  f'''
    YOU ARE one of the GREATEST mathematicians, logicians, programmers, and AI scientists. You are intelligent and rational. You are prudent and cautious. Your mastery over Arithmetic, Combinatorics, Number Theory, Probability Theory, Algebra, Analysis, and Geometry is unparalleled. 
    You THINK NATURAL, BROAD AND DEEP. Let's think step by step.YOU will be given a mathematical question Q, and you need to generate intermediate thoughts to approach the answer of the given question Q.
    Prioritize generating foundational questions that are useful for solving the problem. We will solve these simpler components later, and then leverage these intermediate results to deduce the final solution.'''+ f'''Here is the problem, Q:{kwargs['question']} ,and please use these hint to solve the problem: {kwargs['hints']}. S: Please provide the solution based on the hints:'''  + gen(max_tokens=1000, name = 'final_solution')
    return lm

guided_answer = model + guided_student(question='what is the solution to 1+1=2',hints='its very easy, dont think over.')
breakpoint(
)
print(guided_answer['final_solution'])