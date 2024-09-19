import os
import uuid
import re
import warnings
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from paddleocr import PaddleOCR  # PaddleOCR for image-based text extraction
from io import BytesIO
from PIL import Image
import logging

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl")

source_directory = os.environ.get('SOURCE_DIRECTORY', 'source')

# Set up logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PaddleOCR with GPU support
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

# Define a function to clean the text by removing extra spaces and non-ASCII characters
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

# OCR function for extracting text from images using PaddleOCR
def ocr_image_with_paddle(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))  # Open image from in-memory bytes
        image.save(BytesIO(), format='PNG')  # Ensure format is compatible with OCR
        result = ocr.ocr(image)
        if not result or not result[0]:
            raise ValueError("No text found in image.")
        text = []
        for line in result[0]:
            text.append(line[1][0])
        return "\n".join(text)
    except ValueError as ve:
        logging.warning(f"No valid text found in image: {ve}")
        return ""
    except Exception as e:
        logging.error(f"OCR failed on image: {e}")
        return ""

# Define a function to extract and clean English text from a PDF
def extract_and_clean_english_text(pdf_path, max_page=66):
    pdf_document = fitz.open(pdf_path)
    english_text = []
    for page_num in range(min(len(pdf_document), max_page)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        
        if not text.strip():
            # No text found, perform OCR on the images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    base_image = pdf_document.extract_image(img[0])
                    img_bytes = base_image["image"]
                    
                    # Run PaddleOCR on the in-memory image bytes (no saving to disk)
                    ocr_result = ocr_image_with_paddle(img_bytes)
                    if ocr_result:
                        english_text.append(clean_text(ocr_result))
                    else:
                        logging.info(f"No OCR text found in image on page {page_num}.")
                except Exception as e:
                    logging.error(f"Failed to process image on page {page_num} of {pdf_path}: {e}")
        else:
            # Process the text normally if found
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

# Embedding setup
model_kwargs = {'trust_remote_code': True}
embd = HuggingFaceEmbeddings(model_name="nvidia/NV-Embed-v2", model_kwargs=model_kwargs)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

# Define a function to split text into chunks with metadata
def split_text_into_chunks_with_metadata(text, file_name, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    
    # Create a list of dictionaries, each containing the chunk and metadata (file name)
    return [{'page_content': chunk, 'metadata': {'source': file_name}} for chunk in text_chunks]

if __name__ == "__main__":
    print("Processing the documents, this can take a while! :)")
    
    pdf_dir = source_directory
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # Process each PDF file in the directory
    all_chunks_with_metadata = []
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)  # Get the file name of the PDF
        english_text = extract_and_clean_english_text(pdf_path)

        # Optionally save the cleaned English text to a file
        output_text_file = os.path.join(pdf_dir, f"{os.path.splitext(file_name)[0]}_english.txt")
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(english_text)

        # Split the text into chunks with metadata and add to the combined list
        chunks_with_metadata = split_text_into_chunks_with_metadata(english_text, file_name)
        all_chunks_with_metadata.extend(chunks_with_metadata)
        
    # Separate the text chunks and metadata for storing in Chroma
    texts = [chunk['page_content'] for chunk in all_chunks_with_metadata]
    metadatas = [chunk['metadata'] for chunk in all_chunks_with_metadata]

    # Store the texts with metadata in the Chroma vector store
    vectorstore = Chroma.from_texts(
        texts=texts, 
        metadatas=metadatas,  # Add metadata (source: file name)
        collection_name="rag-chroma", 
        embedding=embd, 
        persist_directory=DB_DIR
    )

    # Persist the vector store to disk
    vectorstore.persist()

    print("Finished processing and storing documents.")