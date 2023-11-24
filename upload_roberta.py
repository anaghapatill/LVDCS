import assemblyai as aai
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import regex as re
import streamlit as st
import os
import io
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import re
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed
import torch


class ytengine:
    def __init__(self, transcript: str) -> None:
        self.qa_model_name = 'deepset/roberta-base-squad2'
        self.qa_model = pipeline('question-answering', model=self.qa_model_name)
        self.sim_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
        self.sim_model = SentenceTransformer(self.sim_model_name)
        self.transcript = transcript

    def ask(self, question_text: str):
        result = self.qa_model(
            question=question_text,
            context=self.transcript,
            doc_stride=256,
            max_answer_len=512,
            max_question_len=128,
        )
        return result['answer']

    def find_similar(self, txt: str, top_k=1):
        txt = self.sim_model.encode(txt)
        embeddings = self.sim_model.encode(self.transcript)
        similarities = util.pytorch_cos_sim(txt, embeddings)
        similarities = similarities.reshape(-1)
        indices = list(torch.argsort(similarities))
        indices = [idx.item() for idx in indices[::-1]][:top_k]
        return indices
    
def run(concatenated_transcript, question):
    model = ytengine(concatenated_transcript)
    answer = model.ask(question)
    similar_indices = model.find_similar(answer)

    similar_contexts = []
    for idx in similar_indices:
        similar_contexts.append(model.transcript.split('.')[idx])

    return answer, similar_contexts

def execute(concatenated_transcript, user_question):

    if user_question:
        answer, similar_contexts = run(concatenated_transcript, user_question)
        return answer        

if __name__ == '__main__':
    execute()