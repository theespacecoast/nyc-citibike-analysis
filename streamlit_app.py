#Import 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from PIL import Image
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

##setup
st.sidebar.title("Citi Bike Dashboard 🚲")
page = st.sidebar.selectbox("Select Page", ["Introduction", "Visualization", "Model Prediction"])  # Add "Automated Report 📑" if using ydata_profiling
image_citibike = Image.open('images/citibike.png')
st.image(image_citibike, width=300)

## Upload all data
root_folder = "CitiBike_Trip_Data"
all_data = []

# Walk through all subdirectories
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all DataFrames
final_df = pd.concat(all_data, ignore_index=True)

## Page 1
if page == "Introduction":
    st.subheader("01 Introduction")
    st.markdown("""
    Welcome to the Citi Bike Explorer Page 🚲\n
    We will explore riding trends through thorough visual analysis and attempt to predict usertype based on age, gender, biking hours, and so much more. In other words, we will try to look for correlations between membership with other user demographics. \n
    This dashboard uses data from 2020. While Citi Bike does present data as recent to 2025 May, they do not provide specific user demographics such as gender and age, which is why we used the dataset from 2020.\n
    """)

    # Preview

    st.markdown("##### 📊 Dataset Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(final_df.head(rows))

    # Dictionary

    st.markdown("##### 📖 Dictionary of Columns")

    feature_info = {
    "tripduration": "Duration of the trip in seconds.",
    "starttime": "Start time and date when the trip began.",
    "stoptime": "End time and date when the trip ended.",
    "start station id": "Unique ID of the station where the trip started.",
    "start station name": "Name of the station where the trip started.",
    "start station latitude": "Latitude coordinate of the start station.",
    "start station longitude": "Longitude coordinate of the start station.",
    "end station id": "Unique ID of the station where the trip ended.",
    "end station name": "Name of the station where the trip ended.",
    "end station latitude": "Latitude coordinate of the end station.",
    "end station longitude": "Longitude coordinate of the end station.",
    "bikeid": "ID of the bicycle used during the trip.",
    "usertype": "Type of user. ('Customer' refers to short-term pass users, while 'Subscriber' refers to annual members.)",
    "birth year": "Year of birth of the rider.",
    "gender": "Gender of the rider. (0 = unknown, 1 = male, 2 = female.)"
    }

    desc_df = pd.DataFrame(feature_info.items(), columns=["Feature", "Description"])

    #Display dicitonary
    st.dataframe(desc_df, use_container_width=True)

    #Summary Statistics
    st.markdown("##### 📖 Summary Statistics")
    st.dataframe(final_df.describe())

    st.markdown("""
    Before we go on to the Visualization page, we want to share the results of two brief graphs that allowed us to decide on the trajectory of our visualization.
    """)

    #Display age, gender
    current_year = 2020  
    df = final_df.copy()

    # Calculate age
    df['age'] = current_year - df['birth year']

    df = df[(df['gender'].isin([1, 2])) & (df['age'].between(15, 90))]

    df['age_group'] = pd.cut(
        df['age'],
        bins=[15, 25, 35, 45, 55, 65, 75, 90],
        labels=['16–25', '26–35', '36–45', '46–55', '56–65', '66–75', '76–90']
    )
    
    st.markdown("##### 👫 Demographics by Age, Gender, and Usertype")

    # Pie chart
    filtered_df = final_df[final_df['gender'].isin([1, 2])].copy()
    gender_map = {1: "Male", 2: "Female"}
    filtered_df['gender_label'] = filtered_df['gender'].map(gender_map)

    filtered_df['group'] = filtered_df['gender_label'] + " " + filtered_df['usertype']

    group_counts = filtered_df['group'].value_counts()

    if not group_counts.empty:
        st.markdown("##### 1️⃣ Pie Chart on Gender and Usertype")

        fig, ax = plt.subplots()
        ax.pie(
            group_counts.values,
            labels=group_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.Pastel1.colors
        )
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("No data available to generate pie chart. Try adjusting filters.")


    # bar chart
    st.markdown("##### 2️⃣ Bar Chart on Age, Gender, and Usertype")
    st.markdown("###### This bar chart visualizes the distribution of male and female Citi Bike riders across age groups, with an optional filter to view patterns by user type.")
    # filter
    usertype_filter = st.multiselect(
        "Select Usertype(s) to Include",
        options=df['usertype'].unique(),
        default=df['usertype'].unique()
    )
    filtered_df = df[df['usertype'].isin(usertype_filter)]

    age_gender_counts = filtered_df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    age_gender_counts.plot(kind='bar', stacked=True, ax=ax1, color=['blue', 'pink'])
    ax1.set_title("Age Group by Gender (Stacked)")
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Number of Riders")
    ax1.legend(title='Gender', labels=['Male (1)', 'Female (2)'])
    st.pyplot(fig1)


 # analysis
    st.markdown("##### 📝 Analysis")
    st.markdown("""
    1️⃣ **Subscribers outnumber short-term (guest) customers**, indicating stronger engagement from long-term users.\n 
    2️⃣ **Male riders are more active than female riders** across both user types.\n 
    3️⃣ **Gender distribution is more balanced among guest customers**, while subscribers are predominantly male.
    """)


## Page 2
elif page == "Visualization":
    st.subheader("02 Data Visualization")


    filtered_df = final_df[final_df['gender'].isin([1, 2])].copy()


    df = final_df.copy()
    df = df[df['gender'].isin([1, 2])]  # Filter out unknown gender
    df['gender_label'] = df['gender'].map({1: 'Male', 2: 'Female'})
    df = df[df['tripduration'] <= 3600]  # Keep trips under 1 hour

    # Plot
    st.subheader("1️⃣ Trip Duration by Gender (Boxplot)")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='gender_label', y='tripduration', ax=ax)
    ax.set_title("Distribution of Trip Durations by Gender")
    ax.set_ylabel("Trip Duration (seconds)")
    ax.set_xlabel("Gender")
    st.pyplot(fig)


    st.subheader("2️⃣ Average Trip Duration by Gender and Usertype")

    # Group by gender and usertype
    avg_duration = filtered_df.groupby(['gender', 'usertype'])['tripduration'].mean().reset_index()

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(data=avg_duration, x='gender', y='tripduration', hue='usertype', ax=ax)

    ax.set_xticklabels(['Male (1)', 'Female (2)'])
    ax.set_ylabel("Average Trip Duration (seconds)")
    ax.set_title("Trip Duration Trends by Gender and Usertype")
    st.pyplot(fig)


    st.subheader("3️⃣ Trip Frequency by Gender and Usertype")

    trip_counts = filtered_df.groupby(['gender', 'usertype']).size().reset_index(name='trip_count')

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(data=trip_counts, x='gender', y='trip_count', hue='usertype', ax=ax)

    ax.set_xticklabels(['Male (1)', 'Female (2)'])
    ax.set_ylabel("Number of Trips")
    ax.set_title("How Often Do Different Groups Ride?")
    st.pyplot(fig)



    st.subheader("4️⃣ Birth Year Density by Usertype (Violin Plot)")

    # Remove outliers
    violin_df = final_df[(final_df['birth year'] > 1920) & (final_df['birth year'] < 2010)]

    # Remove unknowns
    violin_df = violin_df[violin_df['usertype'].notna() & violin_df['birth year'].notna()]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=violin_df, x='usertype', y='birth year', ax=ax)
    ax.set_title("Density of Birth Years by Usertype")
    st.pyplot(fig)


    st.subheader("5️⃣ Time of Day Usage by Age Group")

    time_df = final_df.copy()
    time_df = time_df.dropna(subset=['birth year', 'starttime'])
    time_df['age'] = 2020 - time_df['birth year']
    time_df['age_group'] = pd.cut(time_df['age'], bins=[15, 25, 35, 50, 65, 100], labels=['16–25', '26–35', '36–50', '51–65', '66+'])

    time_df['hour'] = pd.to_datetime(time_df['starttime']).dt.hour

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=time_df, x='hour', hue='age_group', multiple='stack', bins=24)
    ax.set_title("Ride Start Time by Age Group")
    ax.set_xlabel("Hour of Day")
    st.pyplot(fig)



## Page 3
elif page == "Model Prediction":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    st.subheader("03 Model Prediction")

    #Linear Regression Model

    df = final_df.copy()
    df = df[df['gender'].isin([1, 2])]
    df = df.dropna(subset=['birth year', 'tripduration', 'usertype'])
    df['age'] = 2020 - df['birth year']
    df = df[(df['age'] >= 15) & (df['age'] <= 90)]

    df['usertype_encoded'] = df['usertype'].map({'Customer': 0, 'Subscriber': 1})

    X = df[['age', 'gender', 'tripduration']]
    y = df['usertype_encoded']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred_class)

    st.subheader("1️⃣ Linear Regression Model Evaluation")
    st.markdown(f"- **Mean Absolute Error (MAE)**: `{mae:.4f}`")
    st.markdown(f"- **Classification Accuracy (threshold @ 0.5)**: `{accuracy * 100:.2f}%`")

    #Logistic Regression

    df = final_df.copy()
    df = df[df['gender'].isin([1, 2])]
    df['age'] = 2020 - df['birth year']
    df = df[(df['age'] > 10) & (df['age'] < 90)]
    df = df.dropna(subset=['tripduration'])

    df['usertype_encoded'] = LabelEncoder().fit_transform(df['usertype'])  # Subscriber=1, Customer=0

    X = df[['age', 'gender', 'tripduration']]
    y = df['usertype_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.markdown("### 2️⃣ Predicted Usertype Based on User Info")

    #User Interaction
    age_input = st.slider("Select rider's age:", min_value=15, max_value=90, value=30)
    gender_input = st.selectbox("Select rider's gender:", options=["Male", "Female"])
    trip_duration_input = st.slider("Estimated trip duration (in seconds):", min_value=60, max_value=7200, value=900)

    gender_code = 1 if gender_input == "Male" else 2
    input_df = pd.DataFrame([[age_input, gender_code, trip_duration_input]], columns=['age', 'gender', 'tripduration'])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    usertype_label = "Subscriber" if prediction == 1 else "Customer"

    st.markdown(f"### 🚲 Predicted Usertype: **{usertype_label}**")
    st.markdown(f"- Probability of being a Subscriber: **{proba[1]*100:.2f}%**")
    st.markdown(f"- Probability of being a Customer: **{proba[0]*100:.2f}%**")


    




