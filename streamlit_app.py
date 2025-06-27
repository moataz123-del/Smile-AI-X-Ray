import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2

# Set page config
st.set_page_config(
    page_title="Smile X-Ray AI Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .detection-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_objects_on_image(image):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: PIL Image object
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = load_model()
    if model is None:
        return []
    
    try:
        results = model.predict(image)
        result = results[0]
        output = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            prob_percentage = f"{prob * 100:.2f}%"
            output.append([x1, y1, x2, y2, result.names[class_id], prob_percentage])
        
        return output
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return []

def draw_boxes_on_image(image, boxes):
    """
    Draw bounding boxes on the image
    :param image: PIL Image object
    :param boxes: Array of bounding boxes
    :return: PIL Image with boxes drawn
    """
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for box in boxes:
        x1, y1, x2, y2, label, confidence = box
        # Draw rectangle
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Prepare text
        text = f"{label} ({confidence})"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(img_cv, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Convert back to PIL format
    result_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_image)

def main():
    # Header
    st.markdown('<h1 class="main-header">🦷 Smile X-Ray AI Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Minimum confidence level for detections"
        )
        
        st.header("ℹ️ About")
        st.markdown("""
        This application uses YOLOv8 to detect dental conditions in X-ray images.
        
        **How to use:**
        1. Upload an X-ray image
        2. The AI will automatically detect dental conditions
        3. Results will be displayed with bounding boxes
        
        **Supported formats:** JPG, PNG, JPEG
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a dental X-ray image for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display original image
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Detection Results")
            
            # Add a spinner while processing
            with st.spinner("Analyzing image..."):
                # Perform detection
                boxes = detect_objects_on_image(image)
                
                if boxes:
                    # Filter boxes based on confidence threshold
                    filtered_boxes = []
                    for box in boxes:
                        confidence = float(box[5].rstrip('%')) / 100
                        if confidence >= confidence_threshold:
                            filtered_boxes.append(box)
                    
                    if filtered_boxes:
                        # Draw boxes on image
                        result_image = draw_boxes_on_image(image, filtered_boxes)
                        st.image(result_image, caption="Detected Objects", use_column_width=True)
                        
                        # Display detection details
                        st.markdown('<div class="result-section">', unsafe_allow_html=True)
                        st.subheader("📋 Detection Details")
                        
                        for i, box in enumerate(filtered_boxes):
                            x1, y1, x2, y2, label, confidence = box
                            with st.expander(f"Detection {i+1}: {label} ({confidence})"):
                                st.markdown(f"""
                                - **Condition:** {label}
                                - **Confidence:** {confidence}
                                - **Location:** ({x1}, {y1}) to ({x2}, {y2})
                                - **Size:** {x2-x1} × {y2-y1} pixels
                                """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No objects detected with confidence ≥ {confidence_threshold*100:.0f}%")
                else:
                    st.info("No dental conditions detected in this image.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Powered by Smile AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 