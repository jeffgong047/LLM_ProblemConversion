from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# Few Shot Examples
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?"
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?"
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
    # Few shot examples
    few_shot_prompt,
    # New question
    ("user", "{question}"),
])


question_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "was chatgpt around while trump was president?"
question_gen.invoke({"question": question})
'when was ChatGPT developed?'
from langchain.utilities import DuckDuckGoSearchAPIWrapper


search = DuckDuckGoSearchAPIWrapper(max_results=4)

def retriever(query):
    return search.run(query)
retriever(question)


retriever(question_gen.invoke({“question”: question}))

from langchain import hub

response_prompt = hub.pull("langchain-ai/stepback-answer")

chain = {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": question_gen | retriever,
            # Pass on the question
            "question": lambda x: x["question"]
        } | response_prompt | ChatOpenAI(temperature=0) | StrOutputParser()


chain.invoke({"question": question})nvoke({"question": question})