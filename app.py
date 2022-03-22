import logging
import pickle
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.feature_selection import chi2
from utils import seed_everything, INDEX2LEVEL, LEVEL2INDEX
from preprocess import word_tokenize, clean_dialogue


seed_everything(seed=914)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title='eSalesHub Dashboard', layout='wide')


@st.cache
def get_esaleshub_data():
    df = pd.read_csv('./data/esaleshub.csv')
    return df


def tokeniser(text):
    text = clean_dialogue(text)
    return word_tokenize(text)


def main():
    # ---- DATA ----
    # df = get_esaleshub_data()

    # ---- SIDEBAR ----
    st.sidebar.header("Configuration")
    algo = st.sidebar.selectbox(
        'Algorithms: ', 
        ['XGBClassifier', 'LGBMClassifier', 'LogisticRegression'], 
        index=0
    )
    text = st.sidebar.selectbox(
        'Dialogue: ', 
        ['Full', 'Agent', 'Customer', 'Siamese'], 
        index=0
    )

    st.sidebar.write('---')
    st.sidebar.header('Remarks')
    st.sidebar.markdown("""
        There are eight different call types in default to train the selected algorithm.
         - `sales_call_lead`
         - `sales_call_qualified`
         - `sales_call_quote`
         - `sales_call_appointment`
         - `sales_call_sale`
         - `customer_service_call_chase`
         - `customer_service_call_general` 
         - `customer_service_call_cancellation` 
    """)

    try:
        with open(f'./checkpoints/pipeline_{algo.lower()}_{text.lower()}.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    except:
        pipeline = None
    
    # ---- MAINPAGE ----
    st.title("DEMO")
    st.write("""
        This project is to apply natural language processing techniques to 
        categorise the complaint. The challenge in this project is in the 
        cleaning of the data. The transcript of client and customer is extracted 
        from AWS automatic speech recognition (ASR) service, which can convert 
        speech to text. Therefore, this is bound to make mistakes pretty often.
    """)
    st.write('---')

    if text != 'Siamese':
        dialogue = st.text_area(f'Please input {text.lower()} dialogue: ', '', height=200)
    else: 
        left_col, right_col = st.columns([1, 1])
        client_dialogue = left_col.text_area("Please input client's dialogue: ", '', height=200)
        customer_dialogue = right_col.text_area("Please input customer's dialogue: ", '', height=200)

    if st.button('Predict'):
        if text != 'Siamese':
            proba = pipeline.predict_proba([dialogue])[0]
        else:
            tmp = pd.DataFrame({
                'client_dialogue': [client_dialogue], 
                'customer_dialogue': [customer_dialogue]
            })
            proba = pipeline.predict_proba(tmp)[0]
        top_3_idx = np.argsort(proba)[-3:]
        top_3_idx = top_3_idx[::-1]
        top_3_values = [proba[i] for i in top_3_idx]
        top_3_types = [INDEX2LEVEL[i] for i in top_3_idx]
        proba_df = pd.DataFrame({
            'call_types': LEVEL2INDEX.keys(), 
            'proba': proba
        })

        fig = px.bar(proba_df, y='proba', x='call_types', text_auto='.4f')
        fig.update_layout(
            title={'text': f'{algo}', 'xanchor': 'center', 'yanchor': 'top'}, 
            xaxis_title="Call types",
            yaxis_title="Probability", 
            autosize=False,
            height=400
        )
        fig.update_traces(
            textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
        )
        st.plotly_chart(fig, use_container_width=True)

	# ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()