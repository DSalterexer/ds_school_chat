import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import gdown

# твой OpenAI API ключ
os.environ["OPENAI_API_KEY"] = "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"

# ссылка на папку с твоей базой данных
folder_url = "https://drive.google.com/drive/folders/1qE8JyF8pNiNAXsh8naPjqD998ucTNTwz?usp=sharing"

# скачиваем папку с базой (если еще не скачана)
if not os.path.exists("chroma_db"):
    with st.spinner('Загрузка базы данных...'):
        gdown.download_folder(folder_url, output="chroma_db", quiet=False, use_cookies=False)
        st.success('База успешно загружена!')

# загрузка векторной базы
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

# настройка модели QA
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# интерфейс Streamlit
st.title("Чат поддержки пользователей")

question = st.text_input("Задай вопрос:")

if question:
    result = qa({"query": question})
    st.write(result["result"])

    with st.expander("Источники ответа:"):
        for doc in result["source_documents"]:
            st.write(doc.page_content)
