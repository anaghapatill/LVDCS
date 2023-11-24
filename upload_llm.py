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

def save_text_to_pdf(text, pdf_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text, border=0, align="L")
    pdf.output(pdf_filename)

def execute(uploaded_files, user_question):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<HuggingFace API Token>"

    api_key = "<AssemblyAI API Key>"

    load_dotenv()
    
    if uploaded_files is not None:
        
        video_paths = []
        pdfs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                video_paths.append(temp_video_file.name)

        if len(video_paths) == 1:
            st.video(uploaded_file)

        if not os.path.exists('Transcripts'):
                os.makedirs('Transcripts')

        for video_path in video_paths:
           
            aai.settings.api_key = api_key
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(video_path)
            
            pdf_filename = "Transcripts/transcript_" + str(len(pdfs)) + ".pdf"
            save_text_to_pdf(transcript.text, pdf_filename)
            pdfs.append(pdf_filename)

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

                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.2, "max_length": 2048})
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=user_question)
                return response
        
if __name__ == '__main__':
    execute()