import streamlit as st

def show_welcome_page():
    # --- Page Title ---
    st.markdown(
        """
        <div style="text-align:center; padding: 20px;">
            <h1 style="color:#2E86C1; font-size:48px;"> Welcome to <span style="color:#1ABC9C;">MLCAN AI</span></h1>
            <h3 style="color:#555;">Smart AI & Machine Learning for Energy and Beyond</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Intro Message ---
    st.markdown(
        """
        ### ① About MLCan AI
        MLCan AI is a **data-driven innovation hub** specializing in applying 
        **machine learning and artificial intelligence** to solve 
        real-world challenges in **energy, geoscience, and engineering**.  

        With tools for **data exploration, model training, prediction, and interpretability**,  
        this platform helps researchers, engineers, and decision-makers unlock insights from complex datasets.  
        """
    )

    # --- Features Section ---
    st.markdown("### ② What You Can Do Here")
    st.markdown(
        """
        - **Explore Data**: Upload or connect datasets and view interactive EDA dashboards.  
        - **Train Models**: Apply state-of-the-art ML algorithms with one click.  
        - **Explain Predictions**: Use SHAP values and feature importance to understand your models.  
        - **Geospatial Tools**: Generate maps, hotspot analyses, and predictive surfaces.  
        - **Streamlined Workflow**: From raw data to actionable insights — all in one place.  
        """
    )

    # --- Divider ---
    st.markdown("---")

    # --- Call to Action ---
    st.markdown(
        """
        ### ③ Get Started
        👉 Use the sidebar to navigate between sections:  
        - **Upload Data**  
        - **Exploration (EDA)**  
        - **Modeling & Training**  
        - **Results & Interpretation**  

        Let's build something **powerful and intelligent** together! ✨
        """
    )

    # --- Footer ---
    st.markdown(
        """
        <div style="text-align:center; color:gray; font-size:14px; margin-top:30px;">
            © 2025 MLCan AI | Innovation at the intersection of Data & Energy
        </div>
        """,
        unsafe_allow_html=True,
    )
