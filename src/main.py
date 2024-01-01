#from langchain.document_loaders import TextLoader
import os
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import AsyncChromiumLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


urls = [
    'https://fosdem.org/2024/',
    'https://fosdem.org/2024/about/',
    'https://fosdem.org/2024/about/sponsors/',
    'https://fosdem.org/2024/archives/',
    'https://fosdem.org/2024/certification/',
    'https://fosdem.org/2024/faq/',
    'https://fosdem.org/2024/fringe/',
    'https://fosdem.org/2024/news/',
    'https://fosdem.org/2024/news/2023-11-08-devrooms-announced/',
    'https://fosdem.org/2024/news/2023-11-20-accepted-stands-fosdem-2024/',
    'https://fosdem.org/2024/news/2023-11-20-call-for-presentations/',
    'https://fosdem.org/2024/practical/',
    'https://fosdem.org/2024/practical/accessibility/',
    'https://fosdem.org/2024/practical/conduct/',
    'https://fosdem.org/2024/practical/covid/',
    'https://fosdem.org/2024/practical/services/',
    'https://fosdem.org/2024/practical/transportation/',
    'https://fosdem.org/2024/schedule/',
    'https://fosdem.org/2024/schedule/events/',
    'https://fosdem.org/2024/schedule/track/lightning_talks/',
    'https://fosdem.org/2024/schedule/tracks/',
    'https://fosdem.org/2024/social/',
    'https://fosdem.org/2024/stands/',
    'https://fosdem.org/2024/support/donate/',
    'https://fosdem.org/2024/volunteer/'
    ]


loader = AsyncChromiumLoader(urls)
docs = loader.load()

#print(docs[0].page_content)

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

### The easy way ###
# split using text splitter
#text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
#docs = text_splitter.split_documents(docs_transformed)

#for doc in docs:
#        print(doc.page_content)
#        print(doc.metadata)
#input("")

### The hard way ###
# split using markdown headers, then again using token splitter
docs = []
for document in docs_transformed:
    metadata = document.metadata
    page_content = document.page_content

    # split page content into chunks using markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_text_splitter.split_text(document.page_content)
    
    # add source metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)
    
    # split again using a token text splitter to be sure lenght is not too long
    token_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=50)
    for chunk in chunks:
        docs.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))

    #for doc in docs:
    #    print(doc.page_content)
    #    print(doc.metadata)


# Store our documents in a vector store https://python.langchain.com/docs/modules/data_connection/vectorstores/
# here we add persistency
if not os.path.exists("./chroma_db"):
    db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./chroma_db")
else:
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())


# Chat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

context = ""

question = "how long a lightning talk should be?"

messages = [
    SystemMessage(
        content=f"You are a helpful assistant that answer the questions of the user using the following context:\n{context}",
    ),
    HumanMessage(
        content=question
    ),
]
#out = chat(messages)

######################################

embedding_vector = OpenAIEmbeddings().embed_query(question)
docs = db.similarity_search_by_vector(embedding_vector)

for doc in docs:
    context += doc.page_content.replace("\n", " ") + "\n\n"

messages = [
    SystemMessage(
        content=f"You are a helpful assistant that answer the questions of the user using the following context:\n{context}",
    ),
    HumanMessage(
        content=question
    ),
]
out = chat(messages)

print("\n")
# print our sources
for doc in docs:
    print(doc.metadata.get("source"))
for doc in docs:
    print(doc.page_content.replace("\n", " ") + "\n")