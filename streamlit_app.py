#Import 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from PIL import Image
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

##setup
st.sidebar.title("Citi Bike Dashboard 🚲")
page = st.sidebar.selectbox("Select Page", ["Introduction", "Visualization", "Model Prediction", "Model Tuning"])  
image_citibike = Image.open('images/citibike.png')
st.image(image_citibike, width=300)

def setup_mlflow():
    """Setup MLflow tracking with DagsHub"""
    # DagsHub MLflow configuration
    mlflow.set_tracking_uri("https://dagshub.com/theespacecoast/citibike-analysis.mlflow")
    
    # Set your DagsHub credentials (you'll need to get these from DagsHub)
    # Option 1: Set environment variables (recommended)
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'theespacecoast'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '1c2b065865ef1c52c9425efe3c8dbd2985588893'

    
    # Option 2: Direct authentication (less secure)
    # mlflow.set_tracking_uri("https://theespacecoast:your_token@dagshub.com/theespacecoast/citibike-analysis.mlflow")
    
    # Set experiment name
    mlflow.set_experiment("citibike-user-prediction")

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

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    
    st.markdown("### 3️⃣ Visual Model Evaluation")
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_score
    
    # Model selection
    selected_model = st.selectbox(
        "Select a classification model to evaluate:", 
        ["Decision Tree", "Random Forest", "Logistic Regression"]
    )

    if selected_model == "Decision Tree":
        st.markdown("#### Decision Tree Classifier")
        
        # Slider for max_depth
        max_depth = st.slider("Select max_depth:", 1, 20, 3, key="dt_depth")
        
        # Create and fit model
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** {accuracy:.4f}")
        with col2:
            precision = precision_score(y_test, y_pred)
            st.markdown(f"**Precision:** {precision:.4f}")
        
        # Cross-validated accuracy
        cv_scores = cross_val_score(model, X, y, cv=5)
        st.markdown(f"**Mean CV Accuracy:** {cv_scores.mean():.4f}")
        
        # Classification Report
        st.markdown("#### 📊 Classification Report")
        report = classification_report(y_test, y_pred, target_names=["Customer", "Subscriber"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Decision Tree Diagram
        st.markdown("#### Decision Tree Diagram")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
        plot_tree(model, feature_names=X.columns, class_names=["Customer", "Subscriber"], filled=True, ax=ax_tree)
        st.pyplot(fig_tree)
        
    elif selected_model == "Random Forest":
        st.markdown("#### Random Forest Classifier")
        
        # Parameters
        n_estimators = st.slider("Number of trees:", 10, 200, 100, key="rf_estimators")
        max_depth = st.slider("Max depth:", 1, 20, 5, key="rf_depth")
        
        # Single model creation and fitting
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** {accuracy:.4f}")
        with col2:
            precision = precision_score(y_test, y_pred)
            st.markdown(f"**Precision:** {precision:.4f}")
        
        # Classification Report
        st.markdown("#### 📊 Classification Report")
        report = classification_report(y_test, y_pred, target_names=["Customer", "Subscriber"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Single Feature Importance plot
        st.markdown("#### 🔍 Feature Importances")
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(data=feature_df, x='Importance', y='Feature', ax=ax)
        st.pyplot(fig)
        
    elif selected_model == "Logistic Regression":
        st.markdown("#### Logistic Regression")
        
        # Single model creation and fitting
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"**Accuracy:** {accuracy:.4f}")
        with col2:
            precision = precision_score(y_test, y_pred)
            st.markdown(f"**Precision:** {precision:.4f}")
        
        # Classification Report
        st.markdown("#### 📊 Classification Report")
        report = classification_report(y_test, y_pred, target_names=["Customer", "Subscriber"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Single Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Customer", "Subscriber"])
        disp.plot(ax=ax)
        st.pyplot(fig)

##Page 4
elif page == "Model Tuning":

    st.subheader("04 Model Tuning & Experiment Tracking")
     # Setup MLflow
    setup_mlflow()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔧 Manual Grid Search", "📊 Browse MLflow Runs", "📈 MLflow Dashboard"])
    
    with tab1:
        st.markdown("#### Manual Grid Search of Decision Tree")
        
        # Prepare data (same as your existing code)
        df = final_df.copy()
        df = df[df['gender'].isin([1, 2])]
        df['age'] = 2020 - df['birth year']
        df = df[(df['age'] > 10) & (df['age'] < 90)]
        df = df.dropna(subset=['tripduration'])
        df['usertype_encoded'] = LabelEncoder().fit_transform(df['usertype'])
        
        X = df[['age', 'gender', 'tripduration']]
        y = df['usertype_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter selection
        max_depth = st.selectbox("Choose max_depth", [3, 5, 10, None])
        
        if st.button("🚀 Train & Log to MLflow"):
            with st.spinner("Training model and logging to MLflow..."):
                try:
                    # Start MLflow run
                    with mlflow.start_run(run_name=f"DecisionTree_depth_{max_depth}"):
                        # Train model
                        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        from sklearn.metrics import recall_score, f1_score, mean_squared_error
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # For regression-like metrics (as shown in your dashboard)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]  # probabilities
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))  # Manual square root
                        mae = mean_absolute_error(y_test, y_pred_proba)
                        r2 = r2_score(y_test, y_pred_proba)
                        
                        # Log parameters
                        mlflow.log_param("max_depth", max_depth)
                        mlflow.log_param("random_state", 42)
                        mlflow.log_param("model_type", "DecisionTree")
                        
                        # Log metrics
                        mlflow.log_metric("accuracy", accuracy)
                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall", recall)
                        mlflow.log_metric("f1_score", f1)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("r2", r2)
                        
                        # Log dataset info
                        mlflow.log_param("train_samples", len(X_train))
                        mlflow.log_param("test_samples", len(X_test))
                        mlflow.log_param("features", str(list(X.columns)))
                        
                        # Note: Model logging not supported by DagsHub
                        # mlflow.sklearn.log_model(model, "decision_tree_model")
                        
                        st.success("✅ Model trained and logged to MLflow!")
                        st.markdown(f"**Accuracy:** {accuracy:.4f}")
                        st.markdown(f"**RMSE:** {rmse:.4f}")
                        st.markdown(f"**MAE:** {mae:.4f}")
                        st.markdown(f"**R²:** {r2:.4f}")
                        
                        st.info("💡 **Note:** DagsHub MLflow tracks parameters and metrics. Model artifacts are not supported.")
                
                except Exception as e:
                    st.error(f"❌ Error during training/logging: {str(e)}")
                    st.markdown("**Troubleshooting:**")
                    st.markdown("- Check your MLflow connection")
                    st.markdown("- Verify your DagsHub credentials")
                    st.markdown("- Some MLflow features may not be supported by DagsHub")
    
    with tab2:
        st.markdown("#### Browse Logged Experiments")
        
        try:
            # Get MLflow client
            client = MlflowClient()
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name("citibike-user-prediction")
            if experiment:
                # Get all runs
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=20
                )
                
                if runs:
                    # Create dataframe of runs
                    runs_data = []
                    for run in runs:
                        run_data = {
                            'run_id': run.info.run_id[:20] + '...',  # Truncate for display
                            'max_depth': run.data.params.get('max_depth', 'N/A'),
                            'rmse': float(run.data.metrics.get('rmse', 0)),
                            'mae': float(run.data.metrics.get('mae', 0)),
                            'r2': float(run.data.metrics.get('r2', 0)),
                            'accuracy': float(run.data.metrics.get('accuracy', 0)),
                            'start_time': run.info.start_time
                        }
                        runs_data.append(run_data)
                    
                    runs_df = pd.DataFrame(runs_data)
                    st.dataframe(runs_df, use_container_width=True)
                    
                    # Best model visualization
                    if len(runs_df) > 0:
                        best_run = runs_df.loc[runs_df['accuracy'].idxmax()]
                        st.markdown("#### 🏆 Best Performing Run")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best Accuracy", f"{best_run['accuracy']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{best_run['rmse']:.4f}")
                        with col3:
                            st.metric("MAE", f"{best_run['mae']:.4f}")
                        with col4:
                            st.metric("R²", f"{best_run['r2']:.4f}")
                else:
                    st.info("No runs found. Train some models first!")
            else:
                st.warning("Experiment not found. Train a model first to create the experiment.")
                
        except Exception as e:
            st.error(f"Error connecting to MLflow: {str(e)}")
            st.info("Make sure you have the correct MLflow tracking URI and credentials set up.")
    
    with tab3:
        st.markdown("#### 📈 DagsHub MLflow Dashboard")
        st.markdown("""
        Below is the live DagsHub MLflow UI for the **citibike-analysis** repo. 
        You can switch between experiments, look at parameter/metrics charts, and dig into individual runs.
        """)
        
        # Create button to open MLflow UI
        dagshub_url = "https://dagshub.com/theespacecoast/citibike-analysis"
        mlflow_url = f"{dagshub_url}.mlflow"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **🔗 Quick Links:**
            - [View MLflow Experiment]({mlflow_url})
            - [DagsHub Repository]({dagshub_url})
            """)
        
        with col2:
            if st.button("🚀 View MLflow Experiment"):
                st.markdown(f"[Open MLflow Dashboard]({mlflow_url})")
        
        # Embed MLflow UI (if possible)
        st.markdown("---")
        st.markdown("💡 **Tip:** Open the MLflow dashboard in a new tab to see your experiments live!")


# elif page == "Explainability":
#     import shap
#     from sklearn.inspection import PartialDependenceDisplay
#     from sklearn.ensemble import RandomForestClassifier

#     df = final_df.copy()
#     df = df[df['gender'].isin([1, 2])]
#     df['age'] = 2020 - df['birth year']
#     df = df[(df['age'] > 15) & (df['age'] < 90)]
#     df = df.dropna(subset=['tripduration'])
#     df['usertype_encoded'] = LabelEncoder().fit_transform(df['usertype'])

#     X = df[['age', 'gender', 'tripduration']]
#     y = df['usertype_encoded']
#     X_sample = X.sample(n=min(1000, len(X)), random_state=42)
#     y_sample = y[X_sample.index]
    
#     # Train model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)

#     # Create SHAP explainer
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)

#     # For binary classification, use class 1 (positive class)
#     if len(shap_values) == 2:
#         shap_values_plot = shap_values[1]
#     else:
#         shap_values_plot = shap_values

#     # 1. Traditional Feature Importance
#     st.markdown("### 📊 Traditional Feature Importance")
#     importance = pd.DataFrame({
#         'Feature': X.columns,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)

#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=importance, x='Importance', y='Feature', ax=ax1)
#     ax1.set_title('Random Forest Feature Importance')
#     st.pyplot(fig1)

#     # 2. SHAP Feature Importance Bar Plot
#     st.markdown("### 🎯 SHAP Feature Importance")
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     shap.plots.bar(shap_values_plot, max_display=len(X.columns), show=False)
#     plt.title('SHAP Feature Importance')
#     st.pyplot(fig2)

#     # 3. SHAP Beeswarm Plot
#     st.markdown("### 🐝 SHAP Beeswarm Plot")
#     st.write("This plot shows the impact of each feature on model predictions. Each dot represents one prediction.")
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     shap.plots.beeswarm(shap_values_plot, max_display=len(X.columns), show=False)
#     plt.title('SHAP Beeswarm Plot - Feature Impact Distribution')
#     st.pyplot(fig3)

#     # 4. SHAP Waterfall Plot for Individual Predictions
#     st.markdown("### 🌊 SHAP Waterfall Plot")
#     st.write("Shows how each feature contributes to a specific prediction")

#     # Let user select an instance or use a random one
#     col1, col2 = st.columns(2)
#     with col1:
#         instance_idx = st.selectbox(
#             "Select instance to explain:",
#             options=range(min(20, len(X))),  # Show first 20 instances
#             index=0
#         )

#     with col2:
#         if st.button("🎲 Random Instance"):
#             instance_idx = np.random.randint(0, len(X))

#     # Create waterfall plot
#     fig4, ax4 = plt.subplots(figsize=(12, 8))
#     shap.plots.waterfall(
#         explainer.expected_value[1] if len(shap_values) == 2 else explainer.expected_value,
#         shap_values_plot[instance_idx],
#         X.iloc[instance_idx],
#         max_display=len(X.columns),
#         show=False
#     )
#     plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
#     st.pyplot(fig4)

#     # 5. SHAP Summary Statistics
#     st.markdown("### 📈 SHAP Summary Statistics")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric(
#             "Most Important Feature",
#             X.columns[np.argmax(np.abs(shap_values_plot).mean(0))]
#         )

#     with col2:
#         st.metric(
#             "Average |SHAP| Value",
#             f"{np.abs(shap_values_plot).mean():.4f}"
#         )

#     # 6. Partial Dependence Plots (keeping your original)
#     st.markdown("### 📉 How Features Affect Predictions")
#     fig5, ax5 = plt.subplots(1, 2, figsize=(15, 5))
#     PartialDependenceDisplay.from_estimator(model, X, ['age'], ax=ax5[0])
#     PartialDependenceDisplay.from_estimator(model, X, ['tripduration'], ax=ax5[1])
#     st.pyplot(fig5)

#     # 7. SHAP Decision Plot (Advanced)
#     st.markdown("### 🔄 SHAP Decision Plot")
#     st.write("Shows the decision path for multiple predictions")
#     fig6, ax6 = plt.subplots(figsize=(12, 8))
#     # Show decision plot for first 10 instances
#     shap.decision_plot(
#         explainer.expected_value[1] if len(shap_values) == 2 else explainer.expected_value,
#         shap_values_plot[:10],
#         X.iloc[:10],
#         show=False
#     )
#     plt.title('SHAP Decision Plot - First 10 Instances')
#     st.pyplot(fig6)
