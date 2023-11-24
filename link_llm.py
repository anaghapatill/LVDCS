import assemblyai as aai
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from fpdf import FPDF
from PyPDF2 import PdfReader
import os
import re
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed

def validate_youtube_link(url: str) -> str:
    yt_regex = r"^(?:https:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([^#\&\?]+)"
    match = re.match(yt_regex, url)
    if match:
        video_id = match.group(1)
        assert len(video_id) == 11, "Invalid YouTube Link"
        return video_id
    else:
        raise Exception("Invalid YouTube Link")

def fetch_transcript(url: str) -> list:
    video_id = validate_youtube_link(url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id)
        return transcript
    except YouTubeRequestFailed:
        raise Exception('YouTube Request Failed, try again later.')

def save_text_to_pdf(text, pdf_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text, border=0, align="L")
    pdf.output(pdf_filename)

def concatenate_transcripts(video_urls: list):
    pdfs = []
    for index, video_url in enumerate(video_urls):
        try:
            transcript = fetch_transcript(video_url)
            text = " ".join(entry['text'] for entry in transcript)
            pdf_filename = f"Transcripts/transcript_{index}.pdf"
            save_text_to_pdf(text, pdf_filename)
            pdfs.append(pdf_filename)
        except Exception as e:
            print(f"Error processing video {video_url}: {e}")
    return pdfs

def execute(yt_link, user_question):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<HuggingFace API Token>"

    load_dotenv()

    video_urls = []
    if not os.path.exists('Transcripts'):
                os.makedirs('Transcripts')
            
    
    if yt_link:
                links = yt_link.split(',')
                for link in links:
                    video_url = link.strip()  
                    if video_url:
                        video_urls.append(video_url)

    pdfs = concatenate_transcripts(video_urls)

    if pdfs:
                all_chunks = []

                for pdf in pdfs:
                    try:
                        pdf_reader = PdfReader(pdf)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    except Exception as e:
                        print(f"Error reading {pdf}: {e}")
                        continue

                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)

                embeddings = HuggingFaceEmbeddings()
                knowledge_base = FAISS.from_texts(all_chunks, embeddings)

                if user_question:
                    docs = knowledge_base.similarity_search(user_question)

                    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.2, "max_length": 1024})
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)
                    return response

if __name__ == '__main__':
    execute()
