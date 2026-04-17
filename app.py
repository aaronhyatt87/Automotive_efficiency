import streamlit as st
import pandas as pd
import joblib
from app_utils import process_image_yolo, setup_rag, prepare_context
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Automotive Efficiency ML", layout="wide")
st.title("Automotive Efficiency Analytics Dashboard")

@st.cache_data
def load_data(_cache_invalidator=1):
    if os.path.exists('vehicle_catalog.csv'):
        # Ensure year and categorical filters are clean
        return pd.read_csv('vehicle_catalog.csv')
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Please run the `analysis.ipynb` notebook first to download data and train our models. Models and CSV must be in the directory.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Vehicle Selector")
    
    # 1. Select Year
    years = sorted(df['year'].unique(), reverse=True)
    selected_year = st.selectbox("Year", years)
    df_yr = df[df['year'] == selected_year]
    
    # 2. Select Make
    makes = sorted(df_yr['make'].unique())
    selected_make = st.selectbox("Make", makes)
    df_mk = df_yr[df_yr['make'] == selected_make]
    
    # 3. Select Model
    models = sorted(df_mk['model'].unique())
    selected_model = st.selectbox("Model", models)
    df_mdl = df_mk[df_mk['model'] == selected_model]
    
    # 4. Handle ambiguous trims sharing Year, Make, Model
    trims = df_mdl['trany'].unique()
    selected_trim = st.selectbox("Transmission / Trim", trims)
    selected_car_df = df_mdl[df_mdl['trany'] == selected_trim].iloc[0]

with col2:
    st.header("Model Predictions vs Actuals")
    
    if os.path.exists('model_reg_mpg.pkl') and os.path.exists('model_reg_co2.pkl'):
        model_mpg = joblib.load('model_reg_mpg.pkl')
        model_co2 = joblib.load('model_reg_co2.pkl')
        
        feature_cols = [
            'year', 'cylinders', 'displ', 'gear_count', 'is_auto', 'is_electrified', 'is_awd',
            'has_turbo', 'has_super', 'has_startstop',
            'make', 'VClass', 'drivetrain_type', 'trans_type', 'fuel_type_group'
        ]
        features = selected_car_df[feature_cols].to_frame().T
        
        pred_mpg = model_mpg.predict(features)[0]
        pred_co2 = model_co2.predict(features)[0]
        
        c1, c2, c3, c4 = st.columns(4)
        
        # Calculate Delta to use delta component in metric
        delta_mpg = pred_mpg - selected_car_df['comb08']
        delta_co2 = pred_co2 - selected_car_df['co2TailpipeGpm']
        
        c1.metric("Predicted MPG", f"{pred_mpg:.1f}", f"{delta_mpg:.1f} (Bias)", delta_color="inverse")
        c2.metric("Measured MPG", f"{selected_car_df['comb08']:.1f}")
        c3.metric("Predicted CO2 (g/mi)", f"{pred_co2:.1f}g", f"{delta_co2:.1f}g (Bias)", delta_color="inverse")
        c4.metric("Measured CO2 (g/mi)", f"{selected_car_df['co2TailpipeGpm']:.1f}g")
        
    else:
        st.warning("Models not found. Run analysis.ipynb first.")


st.divider()
st.header("Computer Vision - YOLO Car Verification")
st.write("Using ultralytics YOLOv8 for confirming a vehicle is present in an image (COCO class ID 2).")

from PIL import Image

uploaded_file = st.file_uploader("Upload an image to verify...", type=["jpg", "jpeg", "png"])

img_source = None
img_name = ""

if uploaded_file is not None:
    img_source = Image.open(uploaded_file)
    img_name = uploaded_file.name
    st.image(img_source, caption="Uploaded Image", use_container_width=True)
else:
    img_path = "sample_image.png"
    if os.path.exists(img_path):
        img_source = img_path
        img_name = img_path
        st.info(f"Using default image: '{img_path}'. You can upload a different image above.")
        st.image(img_path, caption="Default Image", use_container_width=True)
    else:
        st.warning("Please upload an image.")

if img_source is not None:
    # Only process YOLO on button click to save resources
    if st.button(f"Run YOLO Inference on {img_name}"):
        with st.spinner("Processing image via YOLOv8n..."):
            car_detected, res_im = process_image_yolo(img_source)
            
            if car_detected:
                st.success(f"Car Detected! (COCO Class ID 2 found in '{img_name}')")
            else:
                st.error(f"No car detected in '{img_name}'.")
                
            if res_im is not None:
                # use_container_width replaces the deprecated use_column_width in newer streamlit versions
                st.image(res_im, channels="BGR", caption="YOLOv8 Inference Result", use_container_width=True)


st.divider()
st.header("Local AI RAG Agent (Ollama)")
st.write("Query the local Llama3 model constrained to the selected vehicle's domain context.")

question = st.text_input("Ask a question about this vehicle's efficiency or physical details:")
if st.button("Query LLM Agent"):
    context_str = prepare_context(selected_car_df)
    
    with st.expander("Show Retrieved Context Passed to LLM"):
        st.code(context_str)
        
    with st.spinner("Invoking Local Llama3..."):
        try:
            chain = setup_rag("llama3")
            response = chain.invoke({"context": context_str, "question": question})
            st.success("Response Received:")
            st.write(response)
        except Exception as e:
            st.error(f"Error communicating with local LLM. Is Ollama running? Exception details: \n{str(e)}")
