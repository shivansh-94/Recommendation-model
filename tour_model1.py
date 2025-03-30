import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load and cache datasets for better performance
@st.cache_data
def load_data():
    # File paths for datasets
    tourism_db_path = "cleaned_tourism_db.csv"
    attractions_path = "tourism_attractions.csv"

    # Load the datasets
    tourism_db = pd.read_csv(tourism_db_path)
    attractions = pd.read_csv(attractions_path)

    # Merge datasets based on city IDs
    merged_df = pd.merge(tourism_db, attractions, left_on='id', right_on='tourist_place_id', how='left')

    # Combine columns for feature extraction (State + City)
    merged_df["features"] = merged_df["state"] + " " + merged_df["name"]

    # Normalize numeric columns
    scaler = MinMaxScaler()
    merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]] = scaler.fit_transform(
        merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]]
    )

    # Apply TF-IDF vectorization on combined text features
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    text_features = tfidf_vectorizer.fit_transform(merged_df["features"])

    # Combine text and numeric features for training
    combined_features = np.hstack((text_features.toarray(),
                                   merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]].values))

    # Train KNN model for recommendations
    knn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn_model.fit(combined_features)

    return merged_df, knn_model, tfidf_vectorizer, scaler

# Load data and model
merged_df, knn_model, tfidf_vectorizer, scaler = load_data()

# Function to get available destinations for a selected state
def get_destinations_by_state(state):
    return merged_df[merged_df["state"] == state]["name"].unique()

# Function to fetch all attractions for the selected city by matching city ID
def get_city_attractions(state, destination):
    """
    Retrieve all attractions for the selected city by matching its ID.
    """
    # Identify city information from merged dataset
    city_info = merged_df[(merged_df["state"] == state) & (merged_df["name"] == destination)].iloc[0]

    # Filter all attractions by matching tourist_place_id (city's unique ID)
    city_attractions = merged_df[merged_df["id"] == city_info["id"]]

    return city_info, city_attractions

# Streamlit App Interface
st.title("🌍 City-Specific Tourism Recommender (With Airport & Railway Info)")

# User Input: State and City Dropdowns
state = st.selectbox("📍 Select State:", merged_df["state"].unique())
destination_options = get_destinations_by_state(state)
destination = st.selectbox("🏙️ Select City/Destination:", destination_options)

# Recommendation Button
if st.button("🔍 Find Attractions"):
    # Retrieve city overview and related attractions
    city_info, attractions = get_city_attractions(state, destination)

    # Display city overview including nearest airport & railway station
    st.subheader(f"🏖️ Overview of {destination} ({state})")
    st.markdown(f"""
    - 🎉 **Main Festival:** {city_info["main_festival"]}
    - 🍽️ **Cuisine:** {city_info["cuisine"]}
    - 📅 **Best Time to Visit:** {city_info["best_time_to_visit"]}
    - 💰 **Average Entry Fee:** {city_info["entry_fee"] * 100:.2f} USD
    - 🛡️ **Safety Index:** {city_info["safety_index"] * 10:.1f}/10
    - ☀️ **Weather Impact:** {city_info["weather_impact"] * 10:.1f}/10
    - 🌟 **Popularity Score:** {city_info["popularity_score"] * 10:.1f}/10
    - ✈️ **Nearest Airport:** {city_info["nearest_airport"]}
    - 🚆 **Nearest Railway Station:** {city_info["nearest_train_station"]}
    """)

    # Display city-specific attractions
    st.subheader("📸 Attractions in This City:")
    for _, row in attractions.iterrows():
        st.markdown(f"✅ **{row['attraction_name']}**")

# Footer
st.markdown("---")
st.markdown("💼 Built with ❤️ using **Streamlit**")
