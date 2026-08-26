import streamlit as st

def aplicar_estilo():

    st.markdown("""
<style>

.stApp{
    background:#F5F7FB;
}

h1,h2,h3{
    color:#072B7A;
}

div[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:18px;
    border-left:6px solid #F37021;
    box-shadow:0 2px 12px rgba(0,0,0,.08);
}

</style>
""",
unsafe_allow_html=True)
