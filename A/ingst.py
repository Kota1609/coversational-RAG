import os
import warnings
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoModel

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl-cingst")

source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

chunk_size = 4096
chunk_overlap = 128

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    filtered_files.sort()
    filtered_files = filtered_files[0:590]
    print(filtered_files)
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    with open('stored_texts.pkl', 'wb') as file:
        pickle.dump(texts, file)

    return texts

def load_texts():
    with open('stored_texts.pkl', 'rb') as file:
        texts = pickle.load(file)
    return texts

# Create vector database
def create_vector_database():
    model_kwargs = {'trust_remote_code': True}

    embeddings = HuggingFaceEmbeddings(model_name="nvidia/NV-Embed-v2", model_kwargs=model_kwargs)

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma(
        embedding_function=embeddings,
        # documents=process_documents(),
        persist_directory=DB_DIR
    )
    # db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    collection = vector_database.get()
    texts = process_documents([metadata['source'] for metadata in collection['metadatas']])

    print(f"Creating embeddings. May take some minutes...")
#    batch_size = 200   
#    for i in range(0, len(texts), batch_size):
    vector_database.add_documents(texts)

    vector_database.persist()

if __name__ == "__main__":
    create_vector_database()