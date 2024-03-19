import logging
import streamlit as st

# Custom log handler that sends log messages to Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        new_log = self.format(record) + '\n'
        self.logs.append(new_log)
        st.session_state['logs'] = self.logs
        