from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
import os

loader = TextLoader('./introductions/Thermodynamic_Limits_of_Electronic_Systems.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
query1 = "Tell me about George Washington"
qa.run(query1)