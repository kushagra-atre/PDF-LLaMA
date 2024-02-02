from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import tempfile

def create_vector_store(data_dir):
    '''Create a vector store from PDF files'''
    # define what documents to load
    loader = PyPDFLoader(data_dir)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db


def load_llm():
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Adjust GPU usage based on your hardware
    model_path = f"model/llama-2-7b-chat.Q4_K_M.gguf"
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=256,
        n_batch=512,
        max_tokens=2000,
        temperature=0.4,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm


def create_prompt_template():
    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt

@st.cache_data(hash_funcs={tempfile._TemporaryFileWrapper: lambda _: None})
def create_chain(uploaded_file):
    db = create_vector_store(uploaded_file)
    llm = load_llm()
    prompt = create_prompt_template()
    retriever = db.as_retriever(search_kwargs={'k': 2})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=False,
                                        chain_type_kwargs={'prompt': prompt})
    return chain

def query_doc(chain, question):
    return chain({'query':question})['result']


def handle_question_submission(chain):
    if 'question_input' in st.session_state and st.session_state.question_input:
        question = st.session_state.question_input
        answer = query_doc(chain, question)
        st.session_state['chat_history'].append({"question": question, "answer": answer})
        st.session_state.question_input = ""  # Clear the input box

def main():
    st.title("PDF Chatbot with Llama Model")
    st.write("Start chatting now. Optionally, upload a PDF file for context-specific answers.")

    # Initialize or get chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Initialize Llama model outside of document upload check
    llm = load_llm()

    # Document upload
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    context_enhanced = False

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            data_dir = tmp_file.name

        
        context_enhanced = True
        chain = create_chain(data_dir)

    # Chat Interaction
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "What's on your mind?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)    
        helpful_answer = "I am not sure how to respond to that."

            # Check if context is enhanced with a document
        if context_enhanced and chain is not None:
            response = chain({'query': prompt})
            if response and 'result' in response:
                helpful_answer = response['result']
        elif llm is not None:
                # Correctly format the prompt for the Llama model
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            response = llm(formatted_prompt)
            if response:
                helpful_answer = response

        assistant_answer = helpful_answer
        st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
        st.chat_message("assistant").write(assistant_answer)

if __name__ == "__main__":
    main()