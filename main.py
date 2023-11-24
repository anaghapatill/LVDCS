import streamlit as st
import importlib
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

api_key = "<AssemblyAI API Key>"

def save_text_to_pdf(text, pdf_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text, border=0, align="L")
    pdf.output(pdf_filename)

st.title("LVDCS: Lecture Video Doubt Clarification System")
st.sidebar.header("LVDCS: Lecture Video Doubt Clarification System")

option = st.sidebar.selectbox("Choose an option", ["Upload Local Video(s)", "Add YouTube Video Link(s)"])

if option == "Upload Local Video(s)":
    st.subheader("Upload Local Video(s)")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    load_dotenv()

    uploaded_files = st.file_uploader("Upload a video file", type=["mp4"], accept_multiple_files=True)

    def save_text_to_pdf(text, pdf_filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=text, border=0, align="L")
        pdf.output(pdf_filename)

    if uploaded_files is not None:
        
        video_paths = []
        pdfs = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                video_paths.append(temp_video_file.name)

        if len(video_paths) == 1:
            st.video(uploaded_file)
        transcripts = []
        if not os.path.exists('Transcripts'):
                os.makedirs('Transcripts')

        for video_path in video_paths:
           
            aai.settings.api_key = api_key
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(video_path)

            if transcript.text is not None:
                transcripts.append(transcript.text)
            else:
                st.write("Trasncript generation issue. Please retry!")
                break
            
            pdf_filename = "Transcripts/transcript_" + str(len(pdfs)) + ".pdf"
            save_text_to_pdf(transcript.text, pdf_filename)
            pdfs.append(pdf_filename)

        concatenated_transcript = " ".join(transcripts)
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

            user_question = st.text_input("Ask a question:")
            
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.2, "max_length": 2048})
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=user_question)
                l_module_res = response

                r_module_name = "upload_roberta"
                r_module = importlib.import_module(r_module_name)
                r_module_res = r_module.execute(concatenated_transcript, user_question)

                if r_module_res == "I don't know." or l_module_res == "I don't know." or r_module_res == "I don't know" or l_module_res == "I don't know":
                    long_ans = "Insufficient data to answer the given question."
                    short_ans = "Insufficient data to answer the given question."
                else:
                    if (len(r_module_res) >= len(l_module_res)):
                        long_ans = r_module_res
                        short_ans = l_module_res
                    else:
                        short_ans = r_module_res
                        long_ans = l_module_res

                if long_ans or short_ans:
                    answer_type = st.radio("Select Answer Type", ["Long Answer", "Short Answer"])

                    if answer_type == "Long Answer":
                        st.write(long_ans)
                    else:
                        st.write(short_ans)

elif option == "Add YouTube Video Link(s)":
    st.subheader("Add YouTube Video Link(s)")
    
    yt_link = st.text_input("Paste YouTube video links")

    if yt_link and ',' not in yt_link:
            st.video(yt_link)

    if yt_link:
        question = st.text_input("Enter your question")
        if question:
            r_module_name = "link_roberta"
            r_module = importlib.import_module(r_module_name)
            r_module_res = r_module.execute(yt_link,question)

            l_module_name = "link_llm"
            l_module = importlib.import_module(l_module_name)
            l_module_res = l_module.execute(yt_link,question)

        
            if r_module_res == "I don't know." or l_module_res == "I don't know." or r_module_res == "I don't know" or l_module_res == "I don't know":
                    long_ans = "Insufficient data to answer the given question."
                    short_ans = "Insufficient data to answer the given question."
            else:
                if (len(r_module_res) >= len(l_module_res)):
                                long_ans = r_module_res
                                short_ans = l_module_res
                else:
                                short_ans = r_module_res
                                long_ans = l_module_res

            if long_ans or short_ans:
                answer_type = st.radio("Select Answer Type", ["Long Answer", "Short Answer"])

                if answer_type == "Long Answer":
                    st.write(long_ans)
                else:
                    st.write(short_ans)



