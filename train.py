import warnings
import pickle
import swifter
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import seed_everything, INDEX2LEVEL, LEVEL2INDEX
from preprocess import word_tokenize, clean_dialogue


warnings.filterwarnings("ignore")


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


def siamese():
    # algo = 'XGBClassifier'
    # algo = 'LogisticRegression'
    algo = 'LGBMClassifier'

    train_df = pd.read_csv('./data/esaleshub-tr.csv')
    test_df = pd.read_csv('./data/esaleshub-tt.csv')

    train_df['customer_dialogue'] = train_df['customer_dialogue'].swifter.apply(clean_dialogue)
    train_df['client_dialogue'] = train_df['client_dialogue'].swifter.apply(clean_dialogue)
    test_df['customer_dialogue'] = test_df['customer_dialogue'].swifter.apply(clean_dialogue)
    test_df['client_dialogue'] = test_df['client_dialogue'].swifter.apply(clean_dialogue)

    column_transformer = ColumnTransformer(
        [
            ('customer_vect', TfidfVectorizer(sublinear_tf=True), 'customer_dialogue'),
            ('client_vect', TfidfVectorizer(sublinear_tf=True), 'client_dialogue')
        ],
        remainder='drop', n_jobs=-1, verbose=True
    )

    algo_mapping = {
        'XGBClassifier': XGBClassifier(
            n_estimators=200, 
            max_depth=10, 
            n_jobs=-1, 
            eval_metric='mlogloss'
        ), 
        'LGBMClassifier': LGBMClassifier(
            #n_estimators=200, max_depth=10, n_jobs=-1
        ), 
        'LogisticRegression': LogisticRegression(
            max_iter=1000, n_jobs=-1
        )
    }

    pipeline = Pipeline(
        [
            ("vect", column_transformer), 
            ('scaler', MaxAbsScaler()), 
            ("clf", algo_mapping[algo])
        ], 
        verbose=True
    )

    pipeline.fit(train_df, train_df['level_3_id'])
    
    y_test = test_df['level_3_id']
    y_pred = pipeline.predict(test_df)
    y_prob = pipeline.predict_proba(test_df)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    acc3 = metrics.top_k_accuracy_score(y_test, y_prob, k=3)

    print(
        f'Acc: {acc:.4f}\n'
        f'F1: {f1:.4f}\n'
        f'Acc@3: {acc3:.4f}\n'
    )

    with open(f'./checkpoints/pipeline_{algo.lower()}_siamese.pkl', 'wb') as f:
        pickle.dump(pipeline, f)


def main():
    algo = 'LGBMClassifier'
    text = 'Customer'

    df = get_esaleshub_data()

    df['customer_dialogue'] = df['customer_dialogue'].swifter.apply(clean_dialogue)
    df['client_dialogue'] = df['client_dialogue'].swifter.apply(clean_dialogue)
    df['dialogue'] = df['dialogue'].swifter.apply(clean_dialogue)

    algo_mapping = {
        'XGBClassifier': XGBClassifier(
            n_estimators=200, 
            max_depth=10, 
            n_jobs=-1, 
            eval_metric='mlogloss'
        ), 
        'LGBMClassifier': LGBMClassifier(
            n_jobs=-1
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
                # tokenizer=tokeniser
            )),
            ("tfidf", TfidfTransformer(
                sublinear_tf=True, 
                norm='l2'
            )),
            ("clf", algo_mapping[algo])
        ]
    )

    # scores = cross_validate(
    #     pipeline, 
    #     df[text_input[text]].tolist(), 
    #     df['level_3_id'].tolist(), 
    #     cv=10, 
    #     return_train_score=True, 
    #     n_jobs=-1, 
    #     verbose=10
    # )
    # scores_df = pd.DataFrame(scores)
    # print(scores_df)

    # plt.figure(figsize=(10, 5))
    # scores_df[['train_score', 'test_score']].boxplot()
    # plt.show()

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
    seed_everything(seed=914)
    # main()
    siamese()