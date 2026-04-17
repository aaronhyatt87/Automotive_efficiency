import pandas as pd
from ultralytics import YOLO
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def process_image_yolo(image_path, confidence=0.3):
    """
    Given an image path, runs it through the YOLOv8n model
    to check for COCO class ID 2 (which is Car).
    Returns (boolean_detection, visual_array).
    """
    model = YOLO("yolov8n.pt") 
    results = model(image_path, conf=confidence)
    car_detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            # 2 = car in COCO dataset label classes
            if class_id == 2:
                car_detected = True
                break
                
    if len(results) > 0:
        res_im = results[0].plot()
        return car_detected, res_im
    return car_detected, None

def setup_rag(model_name="llama3"):
    """
    Binds the local Ollama LLM to a standardized system prompt
    for RAG workflows.
    """
    llm = Ollama(model=model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert automotive assistant analyzing the dataset.
        Given the following vehicle context:
        {context}
        
        Answer the user's question accurately based only on this context where possible:
        Question: {question}
        """
    )
    chain = prompt | llm
    return chain

def prepare_context(car_dict):
    """
    Takes a single row pandas series for the car constraints and builds text context.
    """
    ctx = f"Vehicle: {car_dict['year']} {car_dict['make']} {car_dict['model']} ({car_dict['trany']}).\n"
    ctx += f"It has {car_dict['cylinders']} cylinders, {car_dict['displ']} L displacement.\n"
    ctx += f"Combined Fuel Economy (MPG): {car_dict['comb08']}.\n"
    ctx += f"Tailpipe CO2 Emissions: {car_dict['co2TailpipeGpm']} g/mi.\n"
    if car_dict['is_electrified'] == 1:
        ctx += "This vehicle is partially or fully electrified.\n"
    return ctx
