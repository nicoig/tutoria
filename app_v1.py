import os
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


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
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


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

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
    st.set_page_config(page_title="TutorIA - Chatea con tu Profe", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("TutorIA - Chatea con tu Profe :books:")
    st.write("Realizar consultas de cualquier tema en tu material de estudio: Haz preguntas como si estuvieras hablando con tu profesor real. TutorIA busca en tus PDFs y proporciona respuestas claras y concisas.")
    
    user_question = st.text_input("Realiza preguntas sobre la asignatura:")
    if st.button('Enviar'):  # Nuevo botón de enviar
        if user_question:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Tus documentos")
        pdf_docs = st.file_uploader("Cargue sus archivos PDF aquí y haga clic en 'Procesar'", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()