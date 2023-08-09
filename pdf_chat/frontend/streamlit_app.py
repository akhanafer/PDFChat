import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.title("PDF Bot")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": 'Dummy System Prompt',
        },
        {"role": "assistant", "content": "Hello, I'm PDF Bot"},
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Authenticate if logged out
if "success_login" not in st.session_state:
    hashed_passwords = stauth.Hasher(["pass"]).generate()

    with open("pdf_chat/auth_config.yaml") as auth_config_file:
        auth_config = yaml.load(auth_config_file, Loader=SafeLoader)
        auth_config["credentials"]["usernames"]["ali"]["password"] = hashed_passwords[
            0
        ]
    authenticator = stauth.Authenticate(
        auth_config["credentials"],
        auth_config["cookie"]["name"],
        auth_config["cookie"]["key"],
        auth_config["cookie"]["expiry_days"],
        auth_config["preauthorized"],
    )

    name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status == False:
        st.error("Invalid username or password, please try again")
    if authentication_status:
        st.session_state.success_login = True
        st.session_state.persona = auth_config["credentials"]["usernames"][username][
            "persona"
        ]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Successfully logged in as {name}. How can I help?",
            }
        )

# React to user input if logged in:
if prompt := st.chat_input(
    "Log in to chat" if "success_login" not in st.session_state else "Chat",
    disabled=(True if "success_login" not in st.session_state else False),
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
            full_response = ""

            # Simulate stream of response with milliseconds delay
            for response in "dummy response, no bot yet".split():
                full_response += f" {response}"
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
