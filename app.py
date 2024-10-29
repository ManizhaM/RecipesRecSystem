import streamlit as st
from gensim.models import Doc2Vec
import pandas as pd

model = Doc2Vec.load("recipe_doc2vec_model.model")

recepies_df = pd.read_csv("sample.csv") 

recepies_inv_mapper = dict(zip(recepies_df.index, recepies_df["name"]))

st.markdown("""
    <style>
    .stApp {
        background: url('https://i3.wp.com/kartinki.pics/uploads/posts/2021-07/1627173060_21-kartinkin-com-p-fon-dlya-menyu-vostochnoi-kukhni-krasivo-23.jpg?ssl=1');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
        color: white;
    }

    h1, h2, h3, h4 {
        font-family: 'Georgia', serif;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    .stTextInput > div > input {
        font-size: 20px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 1px solid #ddd;
        color: #333;
    }

    .card {
        background-color: #f39c12;
        padding: 20px;
        margin: 15px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        color: #fff;
    }

    .card h3 {
        margin-top: 0;
        color: #fff;
    }

    .card p {
        color: #f5f5f5;
    }

    .stButton button {
        background-color: #e67e22;
        color: white;
        border-radius: 10px;
        padding: 10px;
        border: none;
        font-size: 18px;
        cursor: pointer;
    }

    .stButton button:hover {
        background-color: #d35400;
    }
    </style>
""", unsafe_allow_html=True)

def get_similar_recipes(query_text):
    query_doc = model.infer_vector(query_text.split())
    similars = model.dv.most_similar(positive=[query_doc], topn=5)
    similar_recipes = []

    for sim in similars:
        recipe_id = int(sim[0])
        if recipe_id in recepies_inv_mapper:
            recipe_data = recepies_df.iloc[recipe_id]
            similar_recipes.append(recipe_data)

    return similar_recipes

st.title("🍽️ Сервис рекомендаций рецептов")
st.write("Введите запрос (ингредиенты или краткое описание), чтобы получить рекомендации рецептов:")

query_text = st.text_input("Введите запрос")

if st.button("Получить рекомендации"):
    if query_text:
        try:
            similar_recipes = get_similar_recipes(query_text)
            st.write("🍽️ **Похожие рецепты:**")
            cols = st.columns(2) 

            for i, recipe in enumerate(similar_recipes, start=0): 
                with cols[i % 2]:  
                    with st.container():
                        name = recipe["name"]
                        minutes = recipe["minutes"]
                        tags = ", ".join(eval(recipe["tags"]))
                        steps = "\n".join(eval(recipe["steps"]))
                        n_steps = recipe["n_steps"]

                        st.markdown(f"""
                        <div class="card">
                            <h3>{i+1}. {name}</h3>
                            <p><strong>Время приготовления:</strong> {minutes} минут</p>
                            <p><strong>Теги:</strong> {tags}</p>
                            <p><strong>Количество шагов:</strong> {n_steps}</p>
                            <pre><strong>Шаги приготовления:</strong>\n{steps}</pre>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
    else:
        st.warning("Пожалуйста, введите запрос.")