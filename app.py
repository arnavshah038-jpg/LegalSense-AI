import streamlit as st
import requests
import json


st.set_page_config(page_title="LegalTech AI Dashboard", page_icon="⚖️", layout="wide")

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("⚖️ LegalTech AI Contract Analytics Platform")
st.markdown(" Local AI + Qdrant Vector DB ke sath Contract Analysis aur Intelligence Engine")
st.write("-")


col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("📁 Contract Ingestion")
    uploaded_file = st.file_uploader("Apna Legal Contract (PDF) upload here", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Extract Contract"):
            with st.spinner("PDF se text extract ho raha hai aur Llama3 metadata parse kar raha hai..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{FASTAPI_URL}/extract-and-store/", files=files)
                    
                    if response.status_code == 200:
                        st.success("🎉 Contract successfully processed aur Qdrant mein save ho gaya!")
                        metadata = response.json()
                        st.subheader("📋 Extracted Key Metadata")
                        st.json(metadata)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Backend Connection Failed: {e}")

with col2:
    st.header("💬 Ask Anything About Contract (RAG)")
    user_question = st.text_input("Contract se juda koi bhi legal sawal poochhein:", placeholder="e.g., Termination clause kya hai ya contract kab expire ho raha hai?")
    
    if st.button("🔍 Get Legal Answer"):
        if user_question.strip() != "":
            with st.spinner("Qdrant DB se relevant match dhoondh kar AI jawab taiyar kar raha hai..."):
                try:
                    # FIX: Backend /search/ endpoint ko match karne ke liye 'query' payload bheja
                    payload = {"query": user_question, "limit": 3}
                    response = requests.post(f"{FASTAPI_URL}/search/", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        results_list = result.get("results", [])
                        
                        if results_list:
                            st.markdown("### 🤖 AI Legal Response (Top Match):")
                            top_match = results_list[0]
                            
                            # Backend se aaya hua snippet display karein
                            st.info(top_match.get("text_snippet"))
                            
                            # Metadata aur extra details show karne ke liye
                            with st.expander("📄 View Match Source Details"):
                                st.write(f"**Source Document:** {top_match.get('filename')}")
                                st.write(f"**Vector Match Score:** {top_match.get('score')}")
                                st.json(top_match.get("metadata"))
                        else:
                            st.warning("Database mein isse juda koi match nahi mila.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Backend Connection Failed: {e}")
        else:
            st.warning("Please pehle ek valid sawal likhein!")
