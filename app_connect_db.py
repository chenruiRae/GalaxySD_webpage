import os
import random
import uuid
import streamlit as st
import pandas as pd
import psycopg2
import urllib.parse as urlparse
from datetime import datetime

st.set_page_config(page_title="Can AI Dream of Unseen Galaxies?", layout="centered")

st.title("Can AI Dream of Unseen Galaxies?")
st.subheader("Conditional Diffusion Model for Galaxy Morphology Augmentation")
st.markdown("Chenrui Ma, Zechang Sun, Tao Jing, Zheng Cai, Yuan-Sen Ting, Song Huang, Mingyu Li  \nTsinghua University")

# Sidebar navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔭 Intro", "🧠 Method", "🌌 Generation", "🚀 Application", "📚 Citation", "📝 Quiz"])

st.markdown("---")

# PostgreSQL database connection
@st.cache_resource
def init_connection():
    url = st.secrets["db"]["url"]
    result = urlparse.urlparse(url)

    return psycopg2.connect(
        host=result.hostname,
        port=result.port,
        database=result.path[1:],
        user=result.username,
        password=result.password
    )
conn = init_connection()

def create_table():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_answers (
                id SERIAL PRIMARY KEY,
                username TEXT,
                question TEXT,
                image_id TEXT,
                user_answer TEXT,
                is_correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

create_table()

def store_answer(username, question, image_id, user_answer, is_correct):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO quiz_answers (username, question, image_id, user_answer, is_correct)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, question, image_id, user_answer, is_correct))
        conn.commit()


# Tab content
with tab1:
    st.header("🔭 Introduction")
    st.markdown("""
    Machine learning is transforming astronomy, but its effectiveness is often limited by scarce labeled data—especially for rare objects. We propose a conditional diffusion model that generates realistic galaxy images based on specified visual features, using the Galaxy Zoo 2 dataset for training.
    These synthetic images help ML models better generalize, improving classification accuracy and significantly boosting rare object detection. For example, our approach doubled the detection of dust lane galaxies compared to traditional methods. This work shows how generative models can act as “world simulators”, bridging gaps in data and enabling new discoveries in observational astronomy.
 
    **Key Contributions**:
    - A conditional galaxy image generation model based on morphological prompts.
    - Applications to data augmentation and rare galaxy synthesis.
    - Newly found [early-type dust lane catalog](https://zenodo.org/records/15636756) by classifier trained on augmented data.
    """)

    # add quiz guide and leaderboard
    st.markdown("---")
    st.markdown("### 🎮 Take the Galaxy Quiz!")

    st.markdown("""
    Can you spot fake galaxies among real ones? Our quiz lets you challenge your intuition and observational skills.  
    👉 Go to the **Quiz** tab in the last page and test yourself!  

    Join the quiz and see how you rank among other explorers!
    """)

    # ✅ Leaderboard from PostgreSQL
    leaderboard = []
    with conn.cursor() as cur:
    
        cur.execute("""
            SELECT username, COUNT(*) FILTER (WHERE is_correct), COUNT(*) 
            FROM quiz_answers 
            GROUP BY username
        """)

        for row in cur.fetchall():
            name, score, total_q = row
            leaderboard.append((name, score, total_q))

    if leaderboard:
        leaderboard_df = pd.DataFrame(leaderboard, columns=["Name", "Score", "Total"])
        leaderboard_df["Accuracy"] = leaderboard_df["Score"] / leaderboard_df["Total"]
        leaderboard_df = leaderboard_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        st.markdown("### 🏆 Leaderboard (Top 10)")
        st.dataframe(leaderboard_df.head(10))
    
with tab2:
    st.header("🧠 Method")
    st.image("assets/schema.png", caption="Schematic diagram of our model and downstream tasks.", use_container_width=True)
    st.markdown("""
    We use diffusion models to generate high-quality galaxy images based on morphological descriptions. These models work by gradually transforming random noise into realistic images through a learned denoising process.

    We fine-tuned the pretrained Stable Diffusion v1.5 model using galaxy images and their corresponding morphological labels from the [Galaxy Zoo 2 (GZ2)](https://data.galaxyzoo.org) dataset. For better representation and diversity, we preprocess the GZ2 data by well-sampling, balance and random shuffling and now the [preprocessed dataset](https://zenodo.org/records/15669465) is open-sourced. This allows the model to learn how different visual features relate to specific galaxy types.

    During inference, we observed that using text-only conditioning often leads to systematically over-bright images, due to singularities in the diffusion process at the final denoising step. To mitigate this, we introduce randomly selected reference images as control and begin the denoising process from their noised versions. This helps regulate brightness while preserving visual fidelity.

    After training, the model learns the mapping between morphology and visual features, allowing it to act as a “world simulator.” It can generate both realistic and hypothetical galaxies, including rare or unseen combinations such as early-type galaxies with star-forming traits. These synthetic images can enrich downstream model training and support new scientific discoveries.
    
    **👉 GitHub Repo:** [GalaxySD](https://github.com/chenruiRae/GalaxySD)
    """)

with tab3:
    st.header("🌌 Generation Examples")
    st.markdown("Below are examples of generated galaxies conditioned on different morphological features. " \
    "The annotated capital letters represents prompts used to generate simulated galaxies as [Morphological Prompt Legend](#morph-table) table shows.")
    
    st.markdown('<a name="summary"></a>', unsafe_allow_html=True)
    st.image("assets/summary_sdss.png", caption="Comparison of real galaxy images drom SDSS and those generated by our model under various morphology-related text prompts.", use_container_width=True)

    st.markdown('<a name="morph-table"></a>', unsafe_allow_html=True)
    st.markdown("### 🔠 Morphological Prompt Legend")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        | Type | Prompt                                |
        |------|----------------------------------------|
        | A    | cigar-shaped galaxy                   |
        | B    | completely round galaxy               |
        | C    | edge-on galaxy                        |
        | D    | in-between round galaxy              |
        | E    | ring galaxy                           |
        | F    | spiral galaxy with 4 spiral arms     |
        | G    | spiral galaxy with loosely wound 2 arms |
        | H    | spiral galaxy with tightly wound arms |
        | I    | spiral galaxy with just noticeable bulge    |
        """)

    with col2:
        st.markdown("""
        | Type | Prompt                                      |
        |------|----------------------------------------------|
        | J    | spiral galaxy with obvious bulge            |
        | K    | in-between round galaxy, {dust lane:1.3}    |
        | L    | edge-on galaxy, {dust lane:1.3}, with rounded edge-on bulge     |
        | M    | bar-shaped structure in the galaxy center   |
        | N    | edge-on galaxy with rounded bulge           |
        | O    | in-between round galaxy, merger             |
        | P    | spiral galaxy, just noticeable bulge, merger|
        """)
    
    st.markdown("**Want to generate galaxy yourself? [🤗 HuggingFace](https://github.com/repo)**")
    

with tab4:
    st.header("🚀 Applications")
    st.markdown("""
    - **Data Synthesis**: generate realistic and scientific galaxy images given morphological descriptions, especially for rare galaxy types. 
                For example, generate rare realistic early-type dust lane galaxies (D-ETGs) by merging features from early-type galaxies and disky galaxies with dust lane, as shown of the functions of the world simulator.
    """)
    st.image("assets/diagram_D-ETG.png", caption="Generative extrapolation of early-type galaxies with prominent dust lanes (D-ETGs).", use_container_width=True)
    
    st.markdown("""
    - **Enhacing galaxy classifiers**: Improve galaxy morphology classification especially few-shot learning with synthetic data by our model, outperforming baseline and most focal loss case in purity, completeness and F1-score. 
    - **Boosting rare galaxy detection**: The classifier trained on our augmented data can detect positive classes more effectively in GZ. For example, we successfully newly identified 520 missclassified D-ETGs from 239,695 GZ-SDSS dataset and contribute [a D-ETG catalog](https://zenodo.org/records/15636756) for interested readers.
    """)
    st.image("assets/summary_D-ETG.png", caption="Examples of D-ETGs synthesized by our diffusion model (upper panel) and newly identified instances using machine learning models augmented with the synthesized data (lower panel).", use_container_width=True)
    
    
with tab5:
    st.header("📚 Citation")
    st.code("""
@inproceedings{ma2025can,
  title     = {Can AI Dream of Unseen Galaxies? Conditional Diffusion Model for Galaxy Morphology Augmentation},
  author    = {Chenrui Ma, Zechang Sun, Tao Jing et al.},
  booktitle = {ApJS},
  year      = {2025}
}
    """, language="bibtex")
    st.markdown("Feel free to contact us for collaboration!")
    st.markdown("📧 **Contact:** mcr24, szc22, jingt20 [@mails.tsinghua.edu.cn]")


with tab6:
    st.header("📝 Real vs Generated: Quiz Time!")
    col_text, col_button = st.columns([6, 1])
    with col_text:
        st.markdown("Each question contains four galaxy images — one is generated by our model, the other three are real. Can you spot the fake?")
    with col_button:
        restart = st.button("🔄", help="Restart Quiz", key="restart_quiz")
        if restart:
            st.session_state.pop("quiz_questions", None)

    # initialize questions
    if "quiz_questions" not in st.session_state:
        question_root = "assets/quiz_images"
        all_questions = [os.path.join(question_root, q) for q in os.listdir(question_root) if os.path.isdir(os.path.join(question_root, q))]
        selected_questions = random.sample(all_questions, k=5)  # randomly select 5 questions
        question_data = []

        for q_path in selected_questions:
            all_imgs = os.listdir(q_path)
            choices = [{"file": os.path.join(q_path, f), "label": "Generated" if f.endswith(".png") else "Real"} for f in all_imgs if f.endswith((".png", ".jpg"))]
            random.shuffle(choices)
            question_data.append(choices) 

        st.session_state.quiz_questions = question_data

    question_data = st.session_state.quiz_questions

    # UI for each questions
    user_answers = {}
    st.markdown("### 🔍 Pick the **generated** galaxy:")

    # customize CSS styles and adjust the spacing
    st.markdown("""
        <style>
        div[role="radiogroup"] {
            display: flex;
            gap: 130px; /* control the spacing between options */
        }
        </style>
    """, unsafe_allow_html=True)

    for q_idx, choices in enumerate(question_data):
        st.markdown(f"**Question {q_idx + 1}**")
    
        cols = st.columns(len(choices))
        for i, choice in enumerate(choices):
            with cols[i]:
                st.image(choice["file"], use_container_width=True)

        # single selection
        options = list(range(len(choices)))  # [0, 1, 2, 3]
        selected = st.radio(
            "Select a galaxy:",
            options,
            index=None,  
            key=f"q{q_idx}",
            horizontal=True, 
            label_visibility="collapsed" 
        )

        # save user's selection
        user_answers[q_idx] = selected

    # database storage and reading and writing
    username_input = st.text_input("Your name or ID", value="anonymous", key="quiz_username")

    # check whether the user name has been used from the database
    def is_username_taken_db(name):
        if not name.strip():
            return False
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM quiz_answers WHERE username = %s", (name,))
            count = cur.fetchone()[0]
            return count > 0

    if username_input.strip() == "":
        st.warning("Please enter a valid name.")
        username_valid = False
    elif is_username_taken_db(username_input.strip()):
        st.error("⚠️ This username has already been used. Please choose another.")
        username_valid = False
    else:
        st.success("✅ This username is available.")
        username_valid = True

    if st.button("Submit Answers", disabled=not username_valid, key="submit_quiz_button"):
        correct = 0
        total = len(question_data)
        wrong_questions = []

        for q_idx, choices in enumerate(question_data):
            selected_index = user_answers.get(q_idx, None)
            is_correct = False
            truth_index = None
            truth_file = None

            for i, choice in enumerate(choices):
                if choice["label"] == "Generated":
                    truth_index = i
                    truth_file = os.path.basename(choice["file"])
                    break

            if selected_index == truth_index:
                is_correct = True
                correct += 1
            else:
                wrong_questions.append({
                    "question_id": q_idx + 1,
                    "truth_image": choices[truth_index]["file"],
                    "selected_image": choices[selected_index]["file"] if selected_index is not None else None
                })

            selected_file = os.path.basename(choices[selected_index]["file"]) if selected_index is not None else "None"

            # save to database
            store_answer(
                username=username_input.strip(),
                question=f"Q{q_idx + 1}",
                image_id=selected_file,
                user_answer=choices[selected_index]["label"] if selected_index is not None else "None",
                is_correct=is_correct
            )

        st.success(f"You got {correct}/{total} correct!")

        if wrong_questions:
            st.markdown("### ❌ Review your mistakes")
            for item in wrong_questions:
                st.markdown(f"**Question {item['question_id']}**: Your selection is a real observation, the generated galaxy is shown below.")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(item["selected_image"], caption="❌ Your answer", use_container_width=True)
                with col2:
                    st.image(item["truth_image"], caption="✅ Correct answer", use_container_width=True)
        else:
            st.success("🎉 Perfect score! You got all questions correct!")

    # ✅ Leaderboard from PostgreSQL
    leaderboard = []
    with conn.cursor() as cur:
    
        cur.execute("""
            SELECT username, COUNT(*) FILTER (WHERE is_correct), COUNT(*) 
            FROM quiz_answers 
            GROUP BY username
        """)

        for row in cur.fetchall():
            name, score, total_q = row
            leaderboard.append((name, score, total_q))

    if leaderboard:
        leaderboard_df = pd.DataFrame(leaderboard, columns=["Name", "Score", "Total"])
        leaderboard_df["Accuracy"] = leaderboard_df["Score"] / leaderboard_df["Total"]
        leaderboard_df = leaderboard_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        st.markdown("### 🏆 Leaderboard (Top 10)")
        st.dataframe(leaderboard_df.head(10))
