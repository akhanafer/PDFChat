import streamlit as st
import tempfile

from pdf_chat.backend.chatbot import PDFChatMasterBot
from zipfile import ZipFile

st.title("PDF Bot")

download = False

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "Dummy System Prompt",
        },
        {
            "role": "assistant",
            "content": "Hello, I'm PDF Bot, a chatbot built to help you summarize and analyze PDF documents. To get started, please upload a PDF by clicking on 'Browse files' below",
        },
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# File Upload
if "file_uploaded" not in st.session_state:
    uploaded_file = st.file_uploader("Choose a file", type="pdf", key="file_uploader")
    if uploaded_file:
        bytes_data = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(bytes_data)
        st.session_state.bot = PDFChatMasterBot(pdf_path=temp_file.name)
        temp_file.close()
        st.session_state.file_uploaded = True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"{uploaded_file.name} successfully uploaded, you can now ask me questions concerning this document",
            }
        )
        st.experimental_rerun()


# React to user input if file uploaded:
if prompt := st.chat_input(
    "Upload PDF to start chat" if "file_uploaded" not in st.session_state else "Chat",
    disabled=(True if "file_uploaded" not in st.session_state else False),
):
    if prompt.lower().strip() == "logout":
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
    else:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = st.session_state.bot.query(prompt)
            if isinstance(full_response, str):
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Summary created, you can download it by clicking on the Download button",
                    }
                )
                st.session_state.download_button = True
                st.session_state.download_response = full_response
                st.experimental_rerun()
if "download_button" in st.session_state and "downloaded" not in st.session_state:
    download_response = st.session_state.download_response
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_file = open(f"{tmp_dir}/html_diff.html", "w").write(download_response[1])
        pdf_file = open(f"{tmp_dir}/summary.pdf", "wb").write(download_response[0])
        with ZipFile(f"zipped_file.zip", "w") as zip_object:
            zip_object.write(f"{tmp_dir}/html_diff.html", "./html_diff.html")
            zip_object.write(f"{tmp_dir}/summary.pdf", "./summary.pdf")

        with open(f"zipped_file.zip", "rb") as fp:
            st.download_button(
                "Download", data=fp, file_name="summary.zip", mime="application/zip"
            )
            st.session_state.downloaded = True
