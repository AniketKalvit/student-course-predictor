import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------- Step 1: Create dummy dataset ---------------
# You would normally load real data, but let's generate dummy data for now
def create_dummy_data():
    np.random.seed(42)
    data_size = 300

    # Features
    logical_reasoning = np.random.randint(1, 6, size=data_size) # 1-5
    frontend_interest = np.random.choice([0, 1], size=data_size)
    backend_interest = np.random.choice([0, 1], size=data_size)
    testing_interest = np.random.choice([0, 1], size=data_size)
    math_comfort = np.random.randint(1, 6, size=data_size)
    ai_interest = np.random.choice([0, 1], size=data_size)
    coding_experience = np.random.choice([0, 1], size=data_size)
    study_time = np.random.randint(1, 6, size=data_size)

    # Targets
    courses = np.random.choice(['Java Full Stack', 'C# Full Stack', 'Python GenAI', 'QA'], size=data_size)
    skill_levels = np.random.choice(['Poor', 'Average', 'Good'], size=data_size)

    df = pd.DataFrame({
        'logical_reasoning': logical_reasoning,
        'frontend_interest': frontend_interest,
        'backend_interest': backend_interest,
        'testing_interest': testing_interest,
        'math_comfort': math_comfort,
        'ai_interest': ai_interest,
        'coding_experience': coding_experience,
        'study_time': study_time,
        'course': courses,
        'skill_level': skill_levels
    })
    return df

# --------------- Step 2: Train the model ---------------
def train_models(df):
    X = df.drop(['course', 'skill_level'], axis=1)
    y_course = df['course']
    y_skill = df['skill_level']

    X_train, X_test, y_course_train, y_course_test, y_skill_train, y_skill_test = train_test_split(
        X, y_course, y_skill, test_size=0.2, random_state=42
    )

    course_model = RandomForestClassifier(n_estimators=100, random_state=42)
    skill_model = RandomForestClassifier(n_estimators=100, random_state=42)

    course_model.fit(X_train, y_course_train)
    skill_model.fit(X_train, y_skill_train)

    return course_model, skill_model

# --------------- Step 3: Streamlit Web App ---------------
def main():
    st.set_page_config(page_title="Student Course Predictor", page_icon="🎯")
    st.title("🎯 Student Course & Skill Level Predictor")

    st.markdown("Answer the following questions to get a course recommendation and skill level prediction.")

    # Inputs
    logical_reasoning = st.slider("How good are you at logical reasoning?", 1, 5, 3)
    frontend_interest = st.radio("Are you interested in Frontend Development?", ("Yes", "No"))
    backend_interest = st.radio("Are you interested in Backend Development?", ("Yes", "No"))
    testing_interest = st.radio("Are you interested in QA/Testing?", ("Yes", "No"))
    math_comfort = st.slider("How comfortable are you with Math/Statistics?", 1, 5, 3)
    ai_interest = st.radio("Are you interested in Artificial Intelligence?", ("Yes", "No"))
    coding_experience = st.radio("Do you have prior coding experience?", ("Yes", "No"))
    study_time = st.slider("How many hours can you dedicate to study daily?", 1, 5, 3)

    # Convert inputs
    frontend_interest = 1 if frontend_interest == "Yes" else 0
    backend_interest = 1 if backend_interest == "Yes" else 0
    testing_interest = 1 if testing_interest == "Yes" else 0
    ai_interest = 1 if ai_interest == "Yes" else 0
    coding_experience = 1 if coding_experience == "Yes" else 0

    # Collect into a dataframe
    input_data = pd.DataFrame({
        'logical_reasoning': [logical_reasoning],
        'frontend_interest': [frontend_interest],
        'backend_interest': [backend_interest],
        'testing_interest': [testing_interest],
        'math_comfort': [math_comfort],
        'ai_interest': [ai_interest],
        'coding_experience': [coding_experience],
        'study_time': [study_time]
    })

    # Train model on dummy data
    df = create_dummy_data()
    course_model, skill_model = train_models(df)

    if st.button("Predict"):
        predicted_course = course_model.predict(input_data)[0]
        predicted_skill = skill_model.predict(input_data)[0]

        st.success(f"🎓 Recommended Course: **{predicted_course}**")
        st.info(f"📈 Predicted Skill Level: **{predicted_skill}**")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit and Machine Learning")

if __name__ == "__main__":
    main()

