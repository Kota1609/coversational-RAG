import os
import uuid
import re
import warnings
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chainlit as cl
import redis
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl-cingst")

source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')



# Define a function to clean the text by removing extra spaces and non-ASCII characters
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


# Define a function to extract and clean English text from a PDF
def extract_and_clean_english_text(pdf_path, max_page=66):
    pdf_document = fitz.open(pdf_path)
    english_text = []
    for page_num in range(min(len(pdf_document), max_page)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        lines = re.split(r"\n", text)
        for line in lines:
            try:
                if detect(line) == "en":
                    cleaned_line = clean_text(line)
                    if cleaned_line:
                        english_text.append(cleaned_line)
            except:
                continue
    return " ".join(english_text)

model_kwargs = {'trust_remote_code': True}

embd = HuggingFaceEmbeddings(model_name="nvidia/NV-Embed-v2", model_kwargs=model_kwargs)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


# Define a function to split text into chunks
def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


if __name__ == "__main__":
    print("Processing the documents, this can take a while! :)")
    
    pdf_dir = "source_documents"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # Process each PDF file in the directory
    all_text_chunks = []
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        english_text = extract_and_clean_english_text(pdf_path)

        # Optionally save the cleaned English text to a file
        output_text_file = os.path.join(pdf_dir, f"{os.path.splitext(file_name)[0]}_english.txt")
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(english_text)

        # Split the text into chunks and add to the combined list
        text_chunks = split_text_into_chunks(english_text)
        all_text_chunks.extend(text_chunks)
        
    vectorstore = Chroma.from_texts(
        texts=all_text_chunks, collection_name="rag-chroma", embedding=embd, persist_directory=DB_DIR
    )