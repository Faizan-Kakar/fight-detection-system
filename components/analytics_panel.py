import streamlit as st

def render_analytics_panel(incident_count, detection_accuracy="98%", response_time="45s", status_label="🟢 No Fight", status_color="green"):
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Total Incidents", incident_count, "+15%")
    with col_b:
        st.metric("Detection Accuracy", detection_accuracy, "⬆ 2.3%")
    with col_c:
        st.metric("Response Time", response_time, "⬇ 5s faster")
    with col_d:
        st.markdown(
            f"<h4><span style='color: {status_color};'>{status_label}</span></h4>",
            unsafe_allow_html=True
        )
