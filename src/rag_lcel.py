from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Document loader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/
# https://python.langchain.com/docs/integrations/document_loaders/
from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=8fEEbKJoNbU")
documents = loader.load()

# Split documents with text splitter
# https://python.langchain.com/docs/modules/data_connection/document_transformers/
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0,
    length_function=len,
)
chunks = []
for document in documents:
    chunks += (text_splitter.create_documents([document.page_content], [document.metadata]))

# Store our documents in a vector store
# https://python.langchain.com/docs/modules/data_connection/vectorstores/
# db is persistent and can be reused
db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")

# Chat model with stdout streaming output
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

# Create a retriever with our vector store
# Use MultiQueryRetriever
# https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

# Get the template from langchain hub https://smith.langchain.com/hub
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# create a chain using LCEL
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# questions to ask
questions = [
    "What is effective accelerationism?",
    "What is Kardashev scale?",
    "What energy sources could provide the needed energy?",
    "What is the difference between e/acc and effective altruism?",
    "What can you say abou black holes?",
]

# invoke the LCEL chain for each question
for question in questions:
    print(question+"\n")
    result = rag_chain.invoke(question)
    print("\n")