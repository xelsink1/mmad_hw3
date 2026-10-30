# ДЗ-3

Мазитов Тимур, БПИ247. Использован OpenAI Codex (GPT-6).

Python 3.12.14. Из папки `homework/hw_3`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r wine_app/requirements.txt
streamlit run wine_app/app.py
```

Для ноутбука: `pip install -r wine_app/requirements-notebook.txt`.
Запускать его из `hw_3`. Он сохраняет модель, порог и test.csv.

Модель загружается через `st.cache_resource`, чтобы использовать один объект без повторной загрузки. Таблицы и вероятности — через `st.cache_data`, который сохраняет результаты и отдаёт копии данных. Без декоратора `joblib.load` читал бы модель заново при каждом действии пользователя. Порог не входит в аргументы `test_proba`, поэтому при его изменении вероятности на test не пересчитываются.

Для деплоя запушить папку `wine_app` в GitHub. В Streamlit Community Cloud выбрать репозиторий, файл `homework/hw_3/wine_app/app.py` и Python 3.12, затем Deploy. Ссылку добавить в последнюю ячейку ноутбука. Папку `.venv` не пушить.
