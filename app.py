{\rtf1\ansi\ansicpg1251\cocoartf2758
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
from langchain.chains import RetrievalQA\
from langchain_openai import OpenAI\
from langchain.vectorstores import Chroma\
from langchain_openai import OpenAIEmbeddings\
import os\
\
os.environ["OPENAI_API_KEY"] = "\uc0\u1090 \u1074 \u1086 \u1081 _openai_api_\u1082 \u1083 \u1102 \u1095 "\
\
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())\
\
qa = RetrievalQA.from_chain_type(\
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0),\
    chain_type="stuff",\
    retriever=vectordb.as_retriever(search_kwargs=\{"k": 4\}),\
    return_source_documents=True\
)\
\
st.title("\uc0\u1063 \u1072 \u1090  \u1087 \u1086 \u1076 \u1076 \u1077 \u1088 \u1078 \u1082 \u1080  \u1087 \u1086 \u1083 \u1100 \u1079 \u1086 \u1074 \u1072 \u1090 \u1077 \u1083 \u1077 \u1081 ")\
\
question = st.text_input("\uc0\u1047 \u1072 \u1076 \u1072 \u1081  \u1074 \u1086 \u1087 \u1088 \u1086 \u1089 :")\
\
if question:\
    result = qa(\{"query": question\})\
    st.write(result["result"])\
\
    with st.expander("\uc0\u1048 \u1089 \u1090 \u1086 \u1095 \u1085 \u1080 \u1082 \u1080 :"):\
        for doc in result["source_documents"]:\
            st.write(doc.page_content)\
}