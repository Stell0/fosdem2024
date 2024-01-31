from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Document loader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/
# https://python.langchain.com/docs/integrations/document_loaders/
print("Document loader is loading documents...")
from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=8fEEbKJoNbU")
documents = loader.load()

# Split documents with text splitter
# https://python.langchain.com/docs/modules/data_connection/document_transformers/
print("Text splitter is splitting documents...")
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
# (optional) add persist_directory so we can reuse the db without re-creating it
print("Storing documents and embeddings in vector store...")
db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./chroma_db")


print("Ready to ask!\n###########################################\n")
# Chat model with stdout streaming output
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

# Create a retriever with our vector store
# https://python.langchain.com/docs/modules/data_connection/retrievers/
# (optional) return k most similar documents
retriever = db.as_retriever(search_kwargs={"k": 3})

# Create a prompt template https://python.langchain.com/docs/modules/model_io/prompts/
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
                )
            )
    ]
)

# list of questions to ask
questions = [
    "What is effective accelerationism?",
    "What is Kardashev scale?",
    "What energy sources could provide the needed energy?",
    "What is the difference between e/acc and effective altruism?",
    "What can you say abou black holes?",
]

# put everything together
for question in questions:
    print(f"Question: {question}\n")
    # search for similar documents
    docs = retriever.get_relevant_documents(question)
    # create context merging docs together
    context = "\n\n".join(doc.page_content for doc in docs)
    # get valorized prompt from template
    prompt_val = prompt.invoke({"context": context, "question": question})
    # get response from llm
    result = llm(prompt_val.to_messages())
    print("\n#########################################\n")