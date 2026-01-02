# app.py - Save this as lin_reg_streamlit.py and run: streamlit run lin_reg_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Predict house prices based on area using Linear Regression!**")

# Sidebar for inputs
st.sidebar.header("📏 Prediction Inputs")
area = st.sidebar.number_input("Enter Area (sq ft)", min_value=100.0, max_value=10000.0, value=2500.0, step=50.0)

# Load or train model
@st.cache_resource
def load_or_train_model():
    # Sample data from your notebook
    data = {
        'area': [1790, 2030, 2500, 3200],
        'price': [114300, 114200, 150000, 220000]
    }
    df = pd.DataFrame(data)
    
    # Train model
    model = LinearRegression()
    model.fit(df[['area']], df['price'])
    
    # Save coefficients for display
    st.session_state.model = model
    st.session_state.coef = model.coef_[0]
    st.session_state.intercept = model.intercept_
    
    return model, df

model, sample_df = load_or_train_model()

# Main prediction
if st.button("🔮 Predict Price", type="primary"):
    pred_price = model.predict([[area]])[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price", f"${pred_price:,.0f}", f"${pred_price:,.0f}")
    with col2:
        st.metric("Area", f"{area:,} sq ft")
    with col3:
        st.metric("Price per sq ft", f"${model.coef_[0]:.2f}")
    
    st.success(f"🏠 For **{area:,} sq ft**, predicted price is **${pred_price:,.0f}**!")

# Model info
with st.expander("📊 Model Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Slope (coef)", f"${model.coef_[0]:.2f}/sq ft")
        st.metric("Intercept", f"${model.intercept_:,.0f}")
    with col2:
        st.write("**Equation:** `price = {:.2f} × area + {:.0f}`".format(
            model.coef_[0], model.intercept_))
        st.info("✅ Model trained on your original dataset!")

# Visualization
st.header("📈 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Scatter plot with prediction line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample_df['area'], sample_df['price'], color='red', s=100, label='Training Data')
    
    # Prediction line
    x_range = np.linspace(sample_df['area'].min()-500, sample_df['area'].max()+500, 100)
    y_line = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_line, color='blue', linewidth=2, label='Regression Line')
    
    # Prediction point
    ax.scatter([area], [model.predict([[area]])[0]], color='green', s=150, 
               marker='*', label=f'Prediction ({area:,} sq ft)')
    
    ax.set_xlabel('Area (sq ft)')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    # Interactive price vs area
    fig = px.scatter(sample_df, x='area', y='price', 
                     title="Training Data Points",
                     labels={'area': 'Area (sq ft)', 'price': 'Price ($)'},
                     size_max=20)
    fig.add_hline(y=model.intercept_, line_dash="dash", 
                  annotation_text="Intercept")
    st.plotly_chart(fig, use_container_width=True)

# Batch prediction
st.header("📋 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with 'area' column", type=['csv'])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    if 'area' in test_df.columns:
        test_df['predicted_price'] = model.predict(test_df[['area']])
        test_df['price_per_sqft'] = test_df['predicted_price'] / test_df['area']
        
        st.dataframe(test_df.style.highlight_max(axis=0), use_container_width=True)
        
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Predictions",
            data=csv,
            file_name='house_price_predictions.csv',
            mime='text/csv'
        )
    else:
        st.error("❌ CSV must have 'area' column!")

# Save model (from your notebook)
if st.button("💾 Save Model (Joblib)"):
    joblib.dump(model, "house_price_model.joblib")
    st.success("✅ Model saved as 'house_price_model.joblib'!")

if st.button("📥 Load Model (if saved)"):
    try:
        loaded_model = joblib.load("house_price_model.joblib")
        st.success("✅ Model loaded successfully!")
        st.info(f"Loaded coef: ${loaded_model.coef_[0]:.2f}")
    except:
        st.warning("⚠️ No saved model found. Using fresh model.")

# Footer
st.markdown("---")
st.markdown("*Built from your lin_reg.ipynb • Powered by Streamlit & scikit-learn*")
