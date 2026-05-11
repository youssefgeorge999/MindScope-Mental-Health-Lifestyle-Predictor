import streamlit as st
import joblib
import streamlit as st
import joblib
import pandas as pd

# =========================
# Load artifacts
# =========================
model = joblib.load("saved_models/xgb_model.pkl")
feature_columns = joblib.load("saved_models/feature_columns.pkl")

# =========================
# UI
# =========================
st.title("🧠 MindScope - Mental Health Predictor")

st.write("Fill in the required information:")

# =========================
# Inputs
# =========================
Age = st.number_input("Age", 0, 100, 25)
Gender_Male = st.selectbox("Gender (Male=1, Female=0)", [0, 1])

Caffeine = st.number_input("Caffeine Drinks/Day", 0.0)
Close_Friends = st.number_input("Close Friends Count", 0.0)
Hobby_Time = st.number_input("Hobby Time (hours/week)", 0.0)
Screen_Time = st.number_input("Screen Time (hours/day)", 0.0)
Social_Media = st.number_input("Social Media Hours/Day", 0.0)
Work_Hours = st.number_input("Work Hours/Week", 0.0)
Sleep = st.number_input("Sleep Hours/Night", 0.0)

Mood_Swings = st.number_input("Mood Swings")
Concentration_Difficulty = st.number_input("Concentration Difficulty")
Fatigue = st.number_input("Fatigue")
Anxious_Nervous = st.number_input("Anxious Nervous")
Financial_Stress = st.number_input("Financial Stress")
Feeling_Worthless = st.number_input("Feeling Worthless")
Panic_Attacks = st.number_input("Panic Attacks")
Poor_Appetite = st.number_input("Poor Appetite / Overeating")
Irritability = st.number_input("Irritability")
Work_Stress_Level = st.number_input("Work Stress Level")
Discuss_Mental = st.number_input("Discuss Mental Health")
Sleep_Trouble = st.number_input("Sleep Trouble")
Income_Level = st.number_input("Income Level")
Loneliness = st.number_input("Loneliness")
Compulsive_Behavior = st.number_input("Compulsive Behavior")
Work_Life_Balance = st.number_input("Work Life Balance")
Feeling_Sad = st.number_input("Feeling Sad Down")
Diet_Quality = st.number_input("Diet Quality")
Loss_Of_Interest = st.number_input("Loss Of Interest")
Alcohol = st.number_input("Alcohol Frequency")
Social_Support = st.number_input("Social Support")
Feel_Understood = st.number_input("Feel Understood")
Job_Satisfaction = st.number_input("Job Satisfaction")
Obsessive_Thoughts = st.number_input("Obsessive Thoughts")


# =========================
# Predict Button
# =========================
if st.button("Predict"):

    # 1) build empty full schema
    df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # 2) fill raw inputs
    df.loc[0, "Age"] = Age
    df.loc[0, "Gender_Male"] = Gender_Male
    df.loc[0, "Caffeine_Drinks_Day"] = Caffeine
    df.loc[0, "Close_Friends_Count"] = Close_Friends
    df.loc[0, "Hobby_Time_Hours_Week"] = Hobby_Time
    df.loc[0, "Screen_Time_Hours_Day"] = Screen_Time
    df.loc[0, "Social_Media_Hours_Day"] = Social_Media
    df.loc[0, "Work_Hours_Per_Week"] = Work_Hours
    df.loc[0, "Sleep_Hours_Night"] = Sleep

    df.loc[0, "Mood_Swings"] = Mood_Swings
    df.loc[0, "Concentration_Difficulty"] = Concentration_Difficulty
    df.loc[0, "Fatigue"] = Fatigue
    df.loc[0, "Anxious_Nervous"] = Anxious_Nervous
    df.loc[0, "Financial_Stress"] = Financial_Stress
    df.loc[0, "Feeling_Worthless"] = Feeling_Worthless
    df.loc[0, "Panic_Attacks"] = Panic_Attacks
    df.loc[0, "Poor_Appetite_Or_Overeating"] = Poor_Appetite
    df.loc[0, "Irritability"] = Irritability
    df.loc[0, "Work_Stress_Level"] = Work_Stress_Level
    df.loc[0, "Discuss_Mental_Health"] = Discuss_Mental
    df.loc[0, "Sleep_Trouble"] = Sleep_Trouble
    df.loc[0, "Income_Level"] = Income_Level
    df.loc[0, "Loneliness"] = Loneliness
    df.loc[0, "Compulsive_Behavior"] = Compulsive_Behavior
    df.loc[0, "Work_Life_Balance"] = Work_Life_Balance
    df.loc[0, "Feeling_Sad_Down"] = Feeling_Sad
    df.loc[0, "Diet_Quality"] = Diet_Quality
    df.loc[0, "Loss_Of_Interest"] = Loss_Of_Interest
    df.loc[0, "Alcohol_Frequency"] = Alcohol
    df.loc[0, "Social_Support"] = Social_Support
    df.loc[0, "Feel_Understood"] = Feel_Understood
    df.loc[0, "Job_Satisfaction"] = Job_Satisfaction
    df.loc[0, "Obsessive_Thoughts"] = Obsessive_Thoughts

    # 3) feature engineering (exactly like training)
    df["Stress_x_Loneliness"] = df["Work_Stress_Level"] * df["Loneliness"]
    df["Stress_x_Sleep"] = df["Work_Stress_Level"] * df["Sleep_Hours_Night"]
    df["SocialMedia_x_Sleep"] = df["Social_Media_Hours_Day"] * df["Sleep_Hours_Night"]
    df["Lifestyle_Imbalance"] = df["Work_Hours_Per_Week"] - df["Hobby_Time_Hours_Week"]

    df["Stress_Burden_Score"] = (
        df["Work_Stress_Level"] + df["Financial_Stress"] + df["Anxious_Nervous"]
    ) / 3

    df["Exhaustion_Score"] = (df["Fatigue"] + df["Sleep_Trouble"]) / 2

    df["Emotional_Distress_Score"] = (
        df["Feeling_Sad_Down"] + df["Feeling_Worthless"] + df["Loss_Of_Interest"]
    ) / 3

    df["Cognitive_Difficulty_Score"] = (
        df["Concentration_Difficulty"] + df["Obsessive_Thoughts"]
    ) / 2

    df["Social_Isolation_Score"] = (
        df["Loneliness"] + (5 - df["Social_Support"])
    ) / 2

    # 4) ensure correct order (VERY IMPORTANT)
    df = df[feature_columns]

    # 5) prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # 6) output
    st.subheader("Result")

    if pred == 1:
        st.error("🔴 Mental Health Risk Detected")
    else:
        st.success("🟢 No Mental Health Risk")

    st.write("Probability:", round(prob * 100, 2), "%")