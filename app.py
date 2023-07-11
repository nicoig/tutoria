# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/tutoria.git
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# Para eliminar un repo cargado
# git remote remove origin

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

# git push origin main


###############################################################



import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pickle
from htmlTemplates import css, bot_template, user_template
from langchain.prompts.prompt import PromptTemplate
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(filepaths):
    text = ""
    for filepath in filepaths:
        with open(filepath, "rb") as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, vectorstore_file):
    if os.path.exists(vectorstore_file):
        with open(vectorstore_file, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(vectorstore_file, 'wb') as f:
            pickle.dump(vectorstore, f)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_template = """
        Eres un útil asistente profesor de IA. El usuario le da un archivo cuyo contenido está representado por las siguientes piezas de contexto, utilícelas para responder la pregunta al final.
        Si no sabe la respuesta, simplemente diga que no la sabe. NO intente inventar una respuesta.
        Si la pregunta no está relacionada con el contexto, responda cortésmente que está sintonizado para responder solo preguntas relacionadas con el contexto.
        Use tantos detalles como sea posible al responder.

        context: {context}
        =========
        question: {question}
        ======
        """
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="TutorIA - Chatea con tu Profe", page_icon=":books:", layout="wide")  # Añadido layout="wide" para ocultar la barra lateral
    st.write(css, unsafe_allow_html=True)

    if "raw_text" not in st.session_state or "text_chunks" not in st.session_state:
        # Obtener los archivos PDF y procesarlos
        file_directory = 'files'
        filepaths = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith('.pdf')]
        st.session_state.raw_text = get_pdf_text(filepaths)

        # Dividir el texto en fragmentos
        st.session_state.text_chunks = get_text_chunks(st.session_state.raw_text)

    if "vectorstore" not in st.session_state:
        # Crear el vectorstore
        vectorstore_file = 'vectorstore.pkl'
        st.session_state.vectorstore = get_vectorstore(st.session_state.text_chunks, vectorstore_file)

    if "conversation" not in st.session_state:
        # Crear la cadena de conversación
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("TutorIA - Chatea con tu Profe :books:")
    st.write("Realizar consultas de cualquier tema en tu material de estudio: Haz preguntas como si estuvieras hablando con tu profesor real. TutorIA busca en tus PDFs y proporciona respuestas claras y concisas.")

    user_question = st.text_input("Realiza preguntas sobre la asignatura:")
    if st.button('Enviar'):  
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()
