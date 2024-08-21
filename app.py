import streamlit as st
import torch
import pickle
from sentence_transformers import util


# load save files====================================
embeddings=pickle.load(open("models/embeddings.pkl",'rb'))
sentences=pickle.load(open("models/sentences.pkl","rb"))
model=pickle.load(open("models/model.pkl","rb"))

#function ============================================

def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, model.encode(input_paper))
    
    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
                                 
    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])
    
    return papers_list

#create app ===========================================

st.title("Research Papers Recommandation \n System and subject area prediction")
input_papers=st.text_input("Enter paper title ")

if st.button("Recommend"):
    recommandation_papers=recommendation(input_papers)
    st.write(recommandation_papers)
   
