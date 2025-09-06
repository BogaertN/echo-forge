import streamlit as st
from orchestrator import Orchestrator

st.set_page_config(page_title="EchoForge", layout="wide")

orch = Orchestrator()

st.title("EchoForge: Debate and Journal")

enable_tools = st.sidebar.checkbox("Enable Tools", value=False)

user_input = st.text_area("Enter your question or thought:")
if st.button("Run Flow"):
    with st.spinner("Processing..."):
        result = orch.run_flow(user_input)
    st.success("Synthesis:")
    st.write(result)
    st.info("Resonance map updated: resonance_map.html")

# Journal View
st.header("Journal")
entries = orch.journal.search("")
for entry in entries:
    with st.expander(entry[1]):
        st.write(entry[2])

# Resonance Map
st.header("Resonance Map")
if st.button("Generate Map"):
    orch.map.visualize()
    st.write("Map generated: [View Map](resonance_map.html)")
