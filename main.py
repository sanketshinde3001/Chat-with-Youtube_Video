import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
import streamlit as st
import openai
import tiktoken

st.title(':card_file_box: YOUTUBE-GPT')
with st.form("my_form"):
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"]= openai_api_key
    openai.api_key = openai_api_key
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.") 

    ytlink  = st.text_input("Video link format must be like  - https://www.youtube.com/watch?v=xxxxxxxxxx and have Transcription")
    submitted = st.form_submit_button("Submit")

    if submitted and openai_api_key and ytlink:
        st.write("Wait for some time....")
        loader = YoutubeLoader.from_youtube_url(ytlink, add_video_info=True)
        result = loader.load()
        st.write(f"Video Title is - {result[0].metadata['title']}")
        st.write(f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        llm = OpenAI(temperature=0)
        prompt_template = """Write a concise summary of the following:
        {text}
        CONCISE SUMMARY IN ENGLISH:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
        ans = chain({"input_documents": texts}, return_only_outputs=True)
        # st.write(type(ans))
        st.subheader("Intermediate Summery of Video ")  
        for x in ans.get("intermediate_steps"):
            st.write(x)     
        st.subheader("Final Summery of Video ")
        st.write(ans.get("output_text"))
