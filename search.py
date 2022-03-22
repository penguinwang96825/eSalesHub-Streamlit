import gc
import os
import swifter
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from preprocess import word_tokenize, clean_dialogue
from boostrap import BootstrapOutOfBag
warnings.filterwarnings("ignore")


df_tr = pd.read_csv('./data/esaleshub-tr.csv')
df_tt = pd.read_csv('./data/esaleshub-tt.csv')
df_tr['customer_dialogue'] = df_tr['customer_dialogue'].swifter.apply(clean_dialogue)
df_tr['client_dialogue'] = df_tr['client_dialogue'].swifter.apply(clean_dialogue)
df_tt['customer_dialogue'] = df_tt['customer_dialogue'].swifter.apply(clean_dialogue)
df_tt['client_dialogue'] = df_tt['client_dialogue'].swifter.apply(clean_dialogue)

# df = pd.read_csv('./data/esaleshub.csv')
# df['customer_dialogue'] = df['customer_dialogue'].swifter.apply(clean_dialogue)
# df['client_dialogue'] = df['client_dialogue'].swifter.apply(clean_dialogue)


# Objective function to optimise
def objective(params):
    column_transformer = ColumnTransformer(
        [
            ('customer_vect', TfidfVectorizer(sublinear_tf=True), 'customer_dialogue'),
            ('client_vect', TfidfVectorizer(sublinear_tf=True), 'client_dialogue')
        ],
        remainder='drop', n_jobs=-1, verbose=True
    )
    pipeline = Pipeline(
        [
            ("vect", column_transformer), 
            ('scaler', MaxAbsScaler()), 
            ("clf", xgb.XGBClassifier(n_jobs=-1, eval_metric='mlogloss', **params))
        ], 
        verbose=True
    )

    # pipeline.fit(df_tr, df_tr['level_3_id'])
    # y_pred = pipeline.predict(df_tt)
    # score = metrics.f1_score(df_tt['level_3_id'], y_pred, average='weighted')

    scores = cross_val_score(
        pipeline, 
        df_tr, df_tr['level_3_id'], 
        # cv=BootstrapOutOfBag(random_seed=914), 
        cv=3, 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scoring='f1_weighted'
    )
    score = np.mean(scores).item()

    del pipeline
    gc.collect()

    print(f"SCORE: {score}")
    return {'loss': 1-score, 'status': STATUS_OK}


def main():
    space = {
        'n_estimators': 200, 
        'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
        'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' : hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'tree_method': "hist",
        'booster': 'gbtree',
        'nthread': 4, 
        'use_label_encoder': False
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10, 
        trials=trials
    )

    # Best hyperparameters
    print("\n\n\n The best hyperparameters:")
    print(best)

    column_transformer = ColumnTransformer(
        [
            ('customer_vect', TfidfVectorizer(sublinear_tf=True), 'customer_dialogue'),
            ('client_vect', TfidfVectorizer(sublinear_tf=True), 'client_dialogue')
        ],
        remainder='drop', n_jobs=-1, verbose=True
    )
    pipeline = Pipeline(
        [
            ("vect", column_transformer), 
            ('scaler', MaxAbsScaler()), 
            ("clf", xgb.XGBClassifier(n_jobs=-1, eval_metric='mlogloss', **best))
        ], 
        verbose=True
    )
    pipeline.fit(df_tr, df_tr['level_3_id'])

    y_pred = pipeline.predict(df_tt)
    y_prob = pipeline.predict_proba(df_tt)
    y_test = df_tt['level_3_id']
    
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    acc3 = metrics.top_k_accuracy_score(y_test, y_prob, k=3)

    print(
        f'Acc: {acc:.4f}\n'
        f'F1: {f1:.4f}\n'
        f'Acc@3: {acc3:.4f}\n'
    )


if __name__ == '__main__':
    main()