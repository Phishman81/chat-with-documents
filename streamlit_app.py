import os
import tempfile
from getpass import getpass

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreInfo,
    VectorStoreToolkit,
)
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import streamlit as st

# Set the title and subtitle of the app
st.title('🦜🔗 PDF-Chat: Interact with Your PDFs in a Conversational Way')
st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')

# Loading the Pdf file and return a temporary path for it 
st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

temp_file_path = os.getcwd()
while uploaded_file is None:
    x = 1

# Add password input
password = st.text_input("Enter Password", type='password')
correct_password = st.secrets["password"]
if password != correct_password:
    st.error("The password you entered is incorrect. Please try again.")
    st.stop()

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    st.write("Full path of the uploaded file:", temp_file_path)

# Set APIkey for OpenAI Service from secrets.toml
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader(temp_file_path)
# Split pages from pdf 
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

# Create vectorstore info object
vectorstore_info = VectorStoreInfo(
    name="Pdf",
    description=" A pdf file to answer your questions",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content)
