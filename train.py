import warnings
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import seed_everything, INDEX2LEVEL, LEVEL2INDEX
from preprocess import word_tokenize, clean_dialogue


def get_esaleshub_data():
    df = pd.read_csv('./data/esaleshub.csv')
    df = df[
        [
            'dialogue', 'client_dialogue', 'customer_dialogue', 
            'level_3', 'level_2', 'level_3_id', 'level_2_id'
        ]
    ]
    return df


def tokeniser(text):
    text = clean_dialogue(text)
    return word_tokenize(text)


def main():
    algo = 'LogisticRegression'
    text = 'Full'

    df = get_esaleshub_data()

    algo_mapping = {
        'XGBClassifier': XGBClassifier(
            n_estimators=400, 
            max_depth=10, 
            n_jobs=-1, 
            eval_metric='mlogloss'
        ), 
        'LogisticRegression': LogisticRegression(
            max_iter=1000, n_jobs=-1
        )
    }
    text_input = {
        'Full': 'dialogue', 
        'Agent': 'client_dialogue', 
        'Customer': 'customer_dialogue'
    }

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_input[text]].tolist(), df['level_3_id'].tolist(), 
        test_size=0.25, random_state=914, stratify=df['level_3_id']
    )

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(
                min_df=10, 
                encoding='latin-1', 
                ngram_range=(1, 2), 
                stop_words='english', 
                tokenizer=tokeniser
            )),
            ("tfidf", TfidfTransformer(
                sublinear_tf=True, 
                norm='l2'
            )),
            ("clf", algo_mapping[algo]),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    acc3 = metrics.top_k_accuracy_score(y_test, y_prob, k=3)

    print(
        f'Acc: {acc:.4f}\n'
        f'F1: {f1:.4f}\n'
        f'Acc@3: {acc3:.4f}\n'
    )
    with open(f'./checkpoints/pipeline_{algo.lower()}_{text.lower()}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed_everything(seed=914)
    main()