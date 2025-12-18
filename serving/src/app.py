import streamlit as st
import sys
from datetime import datetime
sys.path.append('/'.join(__file__.split('/')[:-2]))

from src.models.user_data import UserData
from src.serving import run_serving_pipeline

st.set_page_config(page_title="Video Recommendation", layout="wide")
st.title("📺 Video Recommendation System")

# Sidebar for user inputs
st.sidebar.header("User Profile")

# Gender selector
gender_options = {"Female": 0, "Male": 1}
gender_display = st.sidebar.selectbox(
    "Gender",
    options=list(gender_options.keys()),
    index=1  # Default to Male
)
gender = gender_options[gender_display]

# Age slider (20-79)
age_actual = st.sidebar.slider(
    "Age",
    min_value=20,
    max_value=79,
    value=50,
    step=1,
    help="Age in years (20-79)"
)
# Normalize age using MinMaxScaler: (age - 20) / (79 - 20)
age = (age_actual - 20) / 59.0

# City ID
city = st.sidebar.number_input(
    "City ID",
    min_value=0,
    value=225,
    step=1
)

# Community type
community_type = st.sidebar.slider(
    "Community Type",
    min_value=0,
    max_value=3,
    value=2,
    step=1
)

# City level
city_level = st.sidebar.slider(
    "City Level",
    min_value=1,
    max_value=6,
    value=2,
    step=1
)

# Price preference (normalized 0-1)
price = st.sidebar.slider(
    "Price Preference (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    help="Normalized price preference between 0 (lowest) and 1 (highest)"
)

# Hour of day
hour = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=13,
    step=1
)

# Day of week
day_options = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
day_display = st.sidebar.selectbox(
    "Day of Week",
    options=list(day_options.keys()),
    index=datetime.now().weekday()  # Default to today
)
day_of_week = day_options[day_display]

# History PIDs (optional)
history_input = st.sidebar.text_input(
    "Watch History PIDs (comma-separated, optional)",
    value="153366",
    help="Enter video PIDs the user has watched, separated by commas"
)

# Parse history PIDs
history_pids = None
if history_input.strip():
    try:
        history_pids = [int(pid.strip()) for pid in history_input.split(",")]
    except ValueError:
        st.sidebar.error("Invalid PIDs format. Please enter numbers separated by commas.")
        history_pids = None

# Get recommendations button
st.sidebar.markdown("---")
if st.sidebar.button("Get Recommendations", key="get_recs", use_container_width=True):
    try:
        # Create user data
        user_data = UserData(
            gender=gender,
            age=age,
            city=city,
            community_type=community_type,
            city_level=city_level,
            price=price,
            hour=hour,
            day_of_week=day_of_week,
            history_pids=history_pids
        )
        
        st.info("🔄 Generating recommendations...")
        
        # Get recommendations
        recommendations = run_serving_pipeline(user_data)
        
        # Display results
        st.subheader("Top Recommendations")
        
        if recommendations:
            for idx, rec in enumerate(recommendations, 1):
                col1, col2, col3, col4 = st.columns([0.8, 1.8, 1.5, 0.8])
                with col1:
                    st.metric("Rank", idx)
                with col2:
                    st.write(f"**Video ID:** {rec.get('video_id', 'N/A')}")
                    st.write(f"**Duration:** {rec.get('duration', 'N/A')} seconds")
                    st.write(f"**Author Fans:** {rec.get('author_fans_count', 0):,.0f}")
                with col3:
                    with st.expander("▶️ Play Video", expanded=False):
                        st.video(f"videos/{rec.get('video_id', '100341')}.mp4")
                with col4:
                    score = rec.get('score', 0)
                    if isinstance(score, str) or score == "DEMOGRAPHIC_RECOMMENDATIONS":
                        score_display = "N/A"
                    else:
                        score_display = f"{float(score):.3f}"
                    st.metric("Score", score_display)
                st.divider()
        else:
            st.warning("No recommendations found.")
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Display user profile summary
with st.expander("📋 User Profile Summary"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Gender:** {gender_display}")
        st.write(f"**Age:** {age_actual} years")
        st.write(f"**City ID:** {city}")
        st.write(f"**Community Type:** {community_type}")
    with col2:
        st.write(f"**City Level:** {city_level}")
        st.write(f"**Price Preference:** {price:.2f}")
        st.write(f"**Hour:** {hour}:00")
        st.write(f"**Day:** {day_display}")
    if history_pids:
        st.write(f"**Watch History:** {', '.join(map(str, history_pids))}")
