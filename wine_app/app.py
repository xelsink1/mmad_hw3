import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = json.loads((APP_DIR / 'config.json').read_text())
test = load_test()
features = config['features']
X_test = test[features]
y_test = test['target']
p = test_proba(model, X_test)

t = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), step=0.001, key='threshold')
st.title('Оценка вина')
tab1, tab2, tab3, tab4 = st.tabs(['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'])

with tab1:
    values = {}
    for name in features:
        if name == 'is_red':
            values[name] = int(st.toggle('Красное вино'))
        else:
            values[name] = st.number_input(name, min_value=float(X_test[name].min()),
                                           max_value=float(X_test[name].max()),
                                           value=float(X_test[name].median()), format='%.5f')
    sample = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(sample)[0, 1])
    st.write('Вероятность хорошего вина:', round(probability, 3))
    st.write('Включить в подборку' if probability >= t else 'Не включать в подборку')

with tab2:
    pred = p >= t
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    st.metric('Precision', round(precision, 3))
    st.metric('Recall', round(recall, 3))
    st.metric('F1', round(f1, 3))

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, p)
    pr, rc, _ = metrics.precision_recall_curve(y_test, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].scatter(np.mean(pred[y_test == 0]), recall, color='orange', label='Текущий порог')
    axes[1].plot(rc, pr)
    axes[1].scatter(recall, precision, color='orange', label='Текущий порог')
    axes[0].set(title='ROC', xlabel='FPR', ylabel='TPR')
    axes[1].set(title='PR', xlabel='Recall', ylabel='Precision')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Если повысить порог, в подборку попадёт меньше вин, часть хороших будет пропущена. '
             'Если снизить порог, найдём больше хороших вин, но добавим и больше посредственных.')

with tab3:
    file = st.file_uploader('CSV с признаками вин', type=['csv'])
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = [name for name in features if name not in data.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif data.empty:
                st.error('Файл пустой')
            else:
                X = data[features].apply(pd.to_numeric)
                if not np.isfinite(X.to_numpy()).all() or not X['is_red'].isin([0, 1]).all():
                    st.error('Нужны числа без пропусков, is_red должен быть 0 или 1')
                else:
                    data['probability'] = model.predict_proba(X)[:, 1]
                    data['prediction'] = (data['probability'] >= t).astype(int)
                    st.dataframe(data)
                    st.download_button('Скачать', data.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')
        except (ValueError, UnicodeError) as error:
            st.error('Не получилось прочитать CSV: ' + str(error))

with tab4:
    coef = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots()
    coef.plot.barh(ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.write('За хорошее вино:', ', '.join(coef[coef > 0].nlargest(3).index))
    st.write('Против:', ', '.join(coef[coef < 0].nsmallest(3).index))
