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


# @st.cache(allow_output_mutation=True)
# def get_esaleshub_data():
#     df = pd.read_csv('./data/esaleshub.csv')
#     df = df[
#         ['dialogue', 'client_dialogue', 'customer_dialogue', 'level_3', 'level_2', 'level_3_id', 'level_2_id']
#     ]
#     return df


# @st.cache
# def chi_feature(pipeline, N=10):
#     df = pd.read_csv('./data/esaleshub-tr.csv')
#     features = pipeline['vect'].transform(df['dialogue'].tolist())
#     features = pipeline['tfidf'].transform(features)
#     labels = df['level_3_id']
#     for level, idx in sorted(LEVEL2INDEX.items()):
#         features_chi2 = chi2(features.toarray(), labels == idx)
#         indices = np.argsort(features_chi2[0])
#         feature_names = np.array(pipeline['vect'].get_feature_names_out())[indices]
#         unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#         bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#         print("#{}:".format(level))
#         print(" - Most correlated unigrams: {}".format(', '.join(unigrams[-N:])))
#         print(" - Most correlated bigrams: {}".format(', '.join(bigrams[-N:])))
#         print('-' * 100)
#     return unigrams[-N:]


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
        ['XGBClassifier', 'LogisticRegression'], 
        index=0
    )
    text = st.sidebar.selectbox(
        'Dialogue: ', 
        ['Full', 'Agent', 'Customer'], 
        index=0
    )

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

    with open(f'./checkpoints/pipeline_{algo.lower()}_{text.lower()}.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    # ---- MAINPAGE ----
    st.title("DEMO")
    dialogue = st.text_area('Please input dialogue: ', '', height=200)

    if st.button('Predict'):
        proba = pipeline.predict_proba([dialogue])[0]
        top_3_idx = np.argsort(proba)[-3:]
        top_3_idx = top_3_idx[::-1]
        top_3_values = [proba[i] for i in top_3_idx]
        top_3_types = [INDEX2LEVEL[i] for i in top_3_idx]
        proba_df = pd.DataFrame({
            'call_types': LEVEL2INDEX.keys(), 
            'proba': proba
        })

        tokens = word_tokenize(dialogue)
        # unigrams = chi_feature(pipeline)

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