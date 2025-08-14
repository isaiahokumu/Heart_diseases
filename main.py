import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import time
import random
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# =========================
# App Config
# =========================
st.set_page_config(page_title="Heart Disease Analysis & Quiz", layout="wide")
st.title("❤️ Heart Disease EDA & Cardiology Quiz")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_cleaned.csv")
    return df

df = load_data()

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📊 EDA", "🎯 Timed Quiz (Kahoot-style)"])

# =========================
# EDA TAB
# =========================
with tab1:
    st.subheader("Exploratory Data Analysis")
    st.write("Dataset Overview:", df.shape)
    st.dataframe(df.head())

    # Disease Distribution
    fig1, ax1 = plt.subplots()
    sns.countplot(x='num', data=df, ax=ax1)
    ax1.set_title("Disease Distribution (num)")
    st.pyplot(fig1)

    # Max Heart Rate vs Age (Altair)
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='age', y='thalch', color='num:N', tooltip=['age', 'thalch', 'num']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # Fasting Blood Sugar Distribution
    fig2, ax2 = plt.subplots()
    sns.countplot(x='fbs', data=df, ax=ax2)
    ax2.set_title("Fasting Blood Sugar Distribution (fbs)")
    st.pyplot(fig2)

    # Major Vessels Distribution
    fig3, ax3 = plt.subplots()
    sns.countplot(x='ca', data=df, ax=ax3)
    ax3.set_title("Number of Major Vessels Distribution (ca)")
    st.pyplot(fig3)

    # Resting Blood Pressure Distribution
    fig4, ax4 = plt.subplots()
    sns.histplot(df['trestbps'].dropna(), kde=True, ax=ax4)
    ax4.set_title("Resting Blood Pressure Distribution (trestbps)")
    st.pyplot(fig4)

# =========================
# TIMED QUIZ TAB
# =========================
with tab2:
    st.subheader("🩺 Timed Cardiology Quiz (Randomized, Difficulty Tiers)")

    st.markdown("""
    **How it works**
    - Choose difficulty (Easy / Medium / Hard) or Mixed.
    - Each question has a countdown timer (Easy:15s, Medium:12s, Hard:10s).
    - Faster correct answers yield higher points. Results saved to `leaderboard.csv` and a PDF summary.
    """)

    # ---------- QUESTION BANK ----------
    QUESTION_BANK = [
        # Easy
        {"q":"What is a healthy resting heart rate for most adults?","choices":["40-60 bpm","60-100 bpm","100-120 bpm","120-140 bpm"],"answer":"60-100 bpm","diff":"Easy"},
        {"q":"Which organ is primarily affected by myocardial infarction?","choices":["Liver","Heart","Lungs","Kidney"],"answer":"Heart","diff":"Easy"},
        {"q":"Is chest pain a common symptom of heart disease?","choices":["Yes","No"],"answer":"Yes","diff":"Easy"},
        {"q":"What does 'BP' commonly stand for?","choices":["Blood Power","Blood Pressure","Body Pressure","Base Pressure"],"answer":"Blood Pressure","diff":"Easy"},
        {"q":"Does exercise-induced angina mean chest pain during exercise?","choices":["Yes","No"],"answer":"Yes","diff":"Easy"},

        # Medium
        {"q":"Which artery is commonly used to measure pulse at the wrist?","choices":["Carotid","Femoral","Radial","Brachial"],"answer":"Radial","diff":"Medium"},
        {"q":"A thalach in the dataset refers to?","choices":["Cholesterol","Max heart rate achieved","Blood sugar","ST depression"],"answer":"Max heart rate achieved","diff":"Medium"},
        {"q":"Which dataset field indicates fasting blood sugar > 120 mg/dl?","choices":["fbs","cp","thal","slope"],"answer":"fbs","diff":"Medium"},
        {"q":"What does 'ca' represent in the dataset?","choices":["Cholesterol","Number of major vessels","Chest pain","Thallium test"],"answer":"Number of major vessels","diff":"Medium"},
        {"q":"Is 'oldpeak' a measure of ST depression induced by exercise?","choices":["Yes","No"],"answer":"Yes","diff":"Medium"},

        # Hard
        {"q":"Which thal value indicates a reversible defect?","choices":["normal","fixed defect","reversible defect"],"answer":"reversible defect","diff":"Hard"},
        {"q":"Which slope of the peak exercise ST segment is associated with worse prognosis?","choices":["upsloping","flat","downsloping"],"answer":"downsloping","diff":"Hard"},
        {"q":"Higher 'oldpeak' values indicate:","choices":["Less ischemia","More ischemia"],"answer":"More ischemia","diff":"Hard"},
        {"q":"A patient with exang = 1 means:","choices":["No angina","Exercise-induced angina"],"answer":"Exercise-induced angina","diff":"Hard"},
        {"q":"Thallium stress test is used to assess:","choices":["ECG rhythm","Myocardial perfusion","Blood pressure","Cholesterol"],"answer":"Myocardial perfusion","diff":"Hard"},
    ]

    # ---------- UI: Settings ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        player_name = st.text_input("Player name", value="", max_chars=40)
    with col2:
        difficulty_choice = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])
    with col3:
        n_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    # scoring params
    base_scores = {"Easy": 100, "Medium": 150, "Hard": 200}
    time_limits = {"Easy": 15, "Medium": 12, "Hard": 10}  # seconds
    penalty_per_sec = 6  # points deducted per second (adjust for balance)

    # ---------- Session state init ----------
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "q_idx" not in st.session_state:
        st.session_state.q_idx = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "q_start_time" not in st.session_state:
        st.session_state.q_start_time = None
    if "results" not in st.session_state:
        st.session_state.results = []
    if "in_progress" not in st.session_state:
        st.session_state.in_progress = False
    if "time_up" not in st.session_state:
        st.session_state.time_up = False

    # ---------- Start Quiz ----------
    def start_quiz():
        # build question list based on difficulty
        bank = QUESTION_BANK.copy()
        if difficulty_choice != "Mixed":
            bank = [q for q in bank if q["diff"] == difficulty_choice]
        random.shuffle(bank)
        st.session_state.quiz_questions = bank[:n_questions]
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.results = []
        st.session_state.in_progress = True
        st.session_state.time_up = False
        st.session_state.q_start_time = time.time()

    if st.button("Start Quiz"):
        if not player_name.strip():
            st.warning("Please enter your name before starting.")
        else:
            start_quiz()

    # ---------- Display current question ----------
    if st.session_state.in_progress and st.session_state.q_idx < len(st.session_state.quiz_questions):
        q_obj = st.session_state.quiz_questions[st.session_state.q_idx]
        q_text = q_obj["q"]
        q_choices = q_obj["choices"]
        q_diff = q_obj["diff"]
        q_base = base_scores[q_diff]
        q_limit = time_limits[q_diff]

        st.markdown(f"**Question {st.session_state.q_idx + 1} / {len(st.session_state.quiz_questions)}**")
        st.markdown(f"**Difficulty:** {q_diff}  —  **Base points:** {q_base}  —  **Time limit:** {q_limit}s")
        st.write(q_text)

        # reset choice key for unique radio per question index
        choice_key = f"choice_{st.session_state.q_idx}"
        user_choice = st.radio("Select an answer:", options=q_choices, key=choice_key)

        # Timer display (remaining seconds)
        if st.session_state.q_start_time is None:
            st.session_state.q_start_time = time.time()
        elapsed = time.time() - st.session_state.q_start_time
        remaining = int(max(0, q_limit - elapsed))
        prog = min(1.0, elapsed / q_limit)
        st.progress(prog)
        st.caption(f"Time remaining: {remaining} seconds")

        # If time elapsed beyond limit, mark time_up
        if elapsed >= q_limit:
            st.session_state.time_up = True

        cola, colb = st.columns(2)
        with cola:
            if st.button("Submit Answer", key=f"submit_{st.session_state.q_idx}"):
                elapsed = time.time() - st.session_state.q_start_time
                correct = (user_choice == q_obj["answer"])
                # if timed out -> zero
                if elapsed >= q_limit:
                    st.warning("⏰ Time's up for this question! No points.")
                    points = 0
                else:
                    if correct:
                        # faster = more points
                        points = max(0, q_base - int(elapsed * penalty_per_sec))
                    else:
                        points = 0

                st.session_state.score += points
                st.session_state.results.append({
                    "question": q_text,
                    "selected": user_choice,
                    "correct_answer": q_obj["answer"],
                    "difficulty": q_diff,
                    "elapsed_sec": round(elapsed, 2),
                    "points": points
                })
                # move to next question
                st.session_state.q_idx += 1
                st.session_state.q_start_time = time.time()
                st.session_state.time_up = False
                # trigger a rerun so UI updates (st.button already causes rerun)
        with colb:
            if st.button("Skip Question", key=f"skip_{st.session_state.q_idx}"):
                # record as skipped (0 points)
                st.session_state.results.append({
                    "question": q_text,
                    "selected": None,
                    "correct_answer": q_obj["answer"],
                    "difficulty": q_diff,
                    "elapsed_sec": round(time.time() - st.session_state.q_start_time, 2),
                    "points": 0
                })
                st.session_state.q_idx += 1
                st.session_state.q_start_time = time.time()
                st.session_state.time_up = False

    # ---------- Quiz finished ----------
    if st.session_state.in_progress and st.session_state.q_idx >= len(st.session_state.quiz_questions):
        st.session_state.in_progress = False
        st.success(f"🎉 Quiz finished! {player_name} scored {st.session_state.score} points.")
        results_df = pd.DataFrame(st.session_state.results)

        # Save to leaderboard CSV
        lb_file = "leaderboard.csv"
        if os.path.exists(lb_file):
            lb = pd.read_csv(lb_file)
        else:
            lb = pd.DataFrame(columns=["Name", "Score", "Date"])
        new_row = {"Name": player_name, "Score": st.session_state.score, "Date": pd.Timestamp.now()}
        lb = pd.concat([lb, pd.DataFrame([new_row])], ignore_index=True)
        lb.to_csv(lb_file, index=False)

        # Save per-player detailed results
        details_file = f"quiz_{player_name.replace(' ','_')}_{int(time.time())}.csv"
        results_df.to_csv(details_file, index=False)

        # Create PDF summary
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, f"Quiz Results - {player_name}", ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 6, f"Total Score: {st.session_state.score}", ln=True)
        pdf.cell(0, 6, f"Questions: {len(st.session_state.quiz_questions)}", ln=True)
        pdf.ln(6)
        for idx, row in results_df.iterrows():
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, f"Q{idx+1}: {row['question']}")
            pdf.set_font("Arial", "", 11)
            sel = str(row['selected']) if pd.notnull(row['selected']) else "SKIPPED"
            pdf.multi_cell(0, 6, f"Selected: {sel}  |  Correct: {row['correct_answer']}  |  Difficulty: {row['difficulty']}  |  Time: {row['elapsed_sec']}s  |  Points: {row['points']}")
            pdf.ln(2)
        pdf_output = f"quiz_summary_{player_name.replace(' ','_')}_{int(time.time())}.pdf"
        pdf.output(pdf_output)

        st.markdown("**Download your results:**")
        st.download_button("Download Detailed CSV", data=open(details_file,"rb"), file_name=os.path.basename(details_file))
        st.download_button("Download PDF Summary", data=open(pdf_output,"rb"), file_name=os.path.basename(pdf_output))

        # Show leaderboard
        st.subheader("🏆 Leaderboard")
        st.dataframe(lb.sort_values("Score", ascending=False).reset_index(drop=True))

        # Reset button
        if st.button("Play Again"):
            # clear session for quiz only
            st.session_state.quiz_questions = []
            st.session_state.q_idx = 0
            st.session_state.score = 0
            st.session_state.q_start_time = None
            st.session_state.results = []
            st.session_state.in_progress = False
            st.session_state.time_up = False

    # If not started yet
    if (not st.session_state.in_progress) and (st.session_state.q_idx == 0) and (not st.session_state.results):
        st.info("Enter name, choose difficulty and number of questions, then click **Start Quiz**.")

# =========================
# End of app
# =========================
