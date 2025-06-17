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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    Welcome to the Citi Bike Explorer 🚲\n
    This dashboard uses data from 2020. (Citi Bike does not provide demographics of their users for more recent years)\n
    We'll explore riding trends through visual analysis and predict usertype based on age, gender, and biking habits.
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





# Bar plot: average birth year by usertype and gender
st.subheader("Average Birth Year by Usertype and Gender")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=filtered_df, x='usertype', y='birth year', hue='gender', ci=None, ax=ax)
ax.set_title("Younger or Older: Birth Year Trends")
ax.set_ylabel("Average Birth Year")
ax.set_xlabel("User Type")
ax.legend(title="Gender (1 = Male, 2 = Female)")
st.pyplot(fig)

st.subheader("Birth Year Distribution by Usertype")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_df, x='usertype', y='birth year', ax=ax)
st.pyplot(fig)

st.subheader("User Type Distribution by Gender")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x='gender', hue='usertype', ax=ax)
ax.set_title("Usertype Breakdown by Gender")
ax.set_xlabel("Gender (1 = Male, 2 = Female)")
st.pyplot(fig)





st.subheader("Average Trip Duration by Gender and Usertype")

# Group by gender and usertype
avg_duration = filtered_df.groupby(['gender', 'usertype'])['tripduration'].mean().reset_index()

# Plot
fig, ax = plt.subplots()
sns.barplot(data=avg_duration, x='gender', y='tripduration', hue='usertype', ax=ax)

ax.set_xticklabels(['Male (1)', 'Female (2)'])
ax.set_ylabel("Average Trip Duration (seconds)")
ax.set_title("Trip Duration Trends by Gender and Usertype")
st.pyplot(fig)


st.subheader("Trip Frequency by Gender and Usertype")

trip_counts = filtered_df.groupby(['gender', 'usertype']).size().reset_index(name='trip_count')

# Plot
fig, ax = plt.subplots()
sns.barplot(data=trip_counts, x='gender', y='trip_count', hue='usertype', ax=ax)

ax.set_xticklabels(['Male (1)', 'Female (2)'])
ax.set_ylabel("Number of Trips")
ax.set_title("How Often Do Different Groups Ride?")
st.pyplot(fig)



st.subheader("Birth Year Density by Usertype (Violin Plot)")

# Optional: Filter out extreme outliers
violin_df = final_df[(final_df['birth year'] > 1920) & (final_df['birth year'] < 2010)]

# Remove unknowns
violin_df = violin_df[violin_df['usertype'].notna() & violin_df['birth year'].notna()]

# Plot violin
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=violin_df, x='usertype', y='birth year', ax=ax)
ax.set_title("Density of Birth Years by Usertype")
st.pyplot(fig)

st.subheader("Trip Duration vs Age")

# Calculate age (assuming data is from 2020)
age_df = final_df.copy()
age_df = age_df.dropna(subset=['birth year', 'tripduration'])
age_df['age'] = 2020 - age_df['birth year']

# Optional: filter out extreme ages
age_df = age_df[(age_df['age'] >= 16) & (age_df['age'] <= 80)]

# Plot scatter with regression line
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=age_df.sample(1000), x='age', y='tripduration', alpha=0.3, hue='usertype', ax=ax)
sns.regplot(data=age_df, x='age', y='tripduration', scatter=False, ax=ax, color='black')
ax.set_title("Trip Duration by Age (with Trend Line)")
ax.set_ylabel("Trip Duration (seconds)")
st.pyplot(fig)

st.subheader("Time of Day Usage by Age Group")

# Preprocessing
time_df = final_df.copy()
time_df = time_df.dropna(subset=['birth year', 'starttime'])
time_df['age'] = 2020 - time_df['birth year']
time_df['age_group'] = pd.cut(time_df['age'], bins=[15, 25, 35, 50, 65, 100], labels=['16–25', '26–35', '36–50', '51–65', '66+'])

# Convert starttime to hour
time_df['hour'] = pd.to_datetime(time_df['starttime']).dt.hour

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=time_df, x='hour', hue='age_group', multiple='stack', bins=24)
ax.set_title("Ride Start Time by Age Group")
ax.set_xlabel("Hour of Day")
st.pyplot(fig)





