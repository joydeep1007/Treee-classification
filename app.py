import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="🌳 Tree Species Classifier",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(90deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .sidebar-info {
        background: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .model-metrics {
        background: linear-gradient(45deg, #FFF3E0 0%, #F3E5F5 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .species-list {
        max-height: 300px;
        overflow-y: auto;
        background: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Define the tree species classes (30 species based on the dataset)
TREE_CLASSES = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili',
    'cactus', 'champa', 'coconut', 'garmalo', 'gulmohor',
    'gunda', 'jamun', 'kanchan', 'kesudo', 'khajur', 'mango',
    'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren',
    'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal',
    'sonmahor', 'sugarcane', 'vad'
]

@st.cache_resource
def load_model():
    """Load the trained tree classification model"""
    try:
        # Load the custom CNN classifier model
        model_path = 'custom_cnn_classifier.h5'
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model, model_path
        else:
            st.error("❌ Custom CNN model file not found! Please ensure 'custom_cnn_classifier.h5' exists in the current directory.")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading custom CNN model: {str(e)}")
        return None, None

def preprocess_image(image, img_size=224):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Resize image
        img_resized = cv2.resize(img_array, (img_size, img_size))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_tree_species(model, image, class_names):
    """Predict the tree species from the uploaded image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [(class_names[idx], predictions[0][idx]) for idx in top_5_indices]
        
        return predicted_class, confidence, top_5_predictions
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None, None

def create_confidence_chart(top_predictions):
    """Create a confidence chart for top predictions"""
    species = [pred[0] for pred in top_predictions]
    confidences = [pred[1] * 100 for pred in top_predictions]
    
    # Create color scale based on confidence
    colors = ['#2E7D32' if conf >= 70 else '#F57C00' if conf >= 50 else '#D32F2F' 
              for conf in confidences]
    
    fig = go.Figure(data=[
        go.Bar(
            x=species,
            y=confidences,
            marker_color=colors,
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="🎯 Prediction Confidence Scores",
        xaxis_title="Tree Species",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        template="plotly_white",
        font=dict(size=12)
    )
    
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>"
    )
    
    return fig

def display_tree_info(species_name):
    """Display information about the predicted tree species"""
    tree_info = {
        'gulmohor': {
            'scientific_name': 'Delonix regia',
            'common_names': 'Flame Tree, Royal Poinciana',
            'characteristics': 'Large, spreading tree with fern-like leaves and bright red-orange flowers',
            'habitat': 'Tropical and subtropical regions',
            'uses': 'Ornamental, shade tree, timber'
        },
        'eucalyptus': {
            'scientific_name': 'Eucalyptus spp.',
            'common_names': 'Gum Tree, Blue Gum',
            'characteristics': 'Fast-growing evergreen with aromatic leaves and distinctive bark',
            'habitat': 'Native to Australia, widely cultivated',
            'uses': 'Timber, paper production, essential oils'
        },
        'ficus': {
            'scientific_name': 'Ficus spp.',
            'common_names': 'Fig Tree, Banyan Tree',
            'characteristics': 'Large trees with broad canopy and distinctive aerial roots',
            'habitat': 'Tropical and subtropical regions',
            'uses': 'Shade tree, religious significance, fruit'
        },
        # Add more species as needed
    }
    
    if species_name.lower() in tree_info:
        info = tree_info[species_name.lower()]
        st.markdown(f"""
        <div class="info-box">
            <h4>🌿 {species_name.title()} Information</h4>
            <p><strong>Scientific Name:</strong> {info['scientific_name']}</p>
            <p><strong>Common Names:</strong> {info['common_names']}</p>
            <p><strong>Characteristics:</strong> {info['characteristics']}</p>
            <p><strong>Habitat:</strong> {info['habitat']}</p>
            <p><strong>Uses:</strong> {info['uses']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <h4>🌿 {species_name.title()}</h4>
            <p>This is a tree species from our classification dataset. For detailed information about this species, 
            please consult botanical references or field guides.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌳 AI-Powered Tree Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a tree to identify its species using deep learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown(f"""
        <div class="sidebar-info">
            <strong>🤖 Model:</strong> Custom CNN<br>
            <strong>📁 File:</strong> {os.path.basename(model_path)}<br>
            <strong>🏷️ Classes:</strong> {len(TREE_CLASSES)} species<br>
            <strong>📐 Input Size:</strong> 224×224 pixels
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🌳 Supported Species")
        with st.expander("View all 30 species"):
            species_df = pd.DataFrame({'Species': sorted(TREE_CLASSES)})
            st.dataframe(species_df, use_container_width=True)
        
        st.markdown("## ℹ️ How to Use")
        st.markdown("""
        1. 📸 Upload a clear image of a tree
        2. 🔍 Wait for AI analysis
        3. 📊 View prediction results
        4. 🌿 Learn about the species
        """)
        
        st.markdown("## 💡 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit images
        - Include leaves and bark when possible
        - Avoid heavily filtered images
        - Single tree per image works best
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Tree Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a tree for species identification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Tree Image", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            **📋 Image Details:**
            - **Filename:** {uploaded_file.name}
            - **Size:** {image.size[0]} × {image.size[1]} pixels
            - **Format:** {image.format}
            """)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 Analysis Results")
            
            # Show loading spinner while processing
            with st.spinner('🧠 AI is analyzing the tree image...'):
                predicted_species, confidence, top_predictions = predict_tree_species(
                    model, image, TREE_CLASSES
                )
            
            if predicted_species is not None:
                # Determine confidence level and color
                if confidence >= 0.7:
                    confidence_class = "confidence-high"
                    confidence_emoji = "🎯"
                elif confidence >= 0.5:
                    confidence_class = "confidence-medium"
                    confidence_emoji = "⚠️"
                else:
                    confidence_class = "confidence-low"
                    confidence_emoji = "❓"
                
                # Display main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{confidence_emoji} Predicted Species</h2>
                    <h1 style="color: #2E7D32; margin: 1rem 0;">{predicted_species.title()}</h1>
                    <p class="{confidence_class}">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Success/warning message based on confidence
                if confidence >= 0.7:
                    st.success(f"✅ High confidence prediction! The AI is {confidence:.1%} confident this is a {predicted_species}.")
                elif confidence >= 0.5:
                    st.warning(f"⚠️ Medium confidence prediction. The AI is {confidence:.1%} confident this is a {predicted_species}.")
                else:
                    st.error(f"❓ Low confidence prediction. The AI is only {confidence:.1%} confident. Consider uploading a clearer image.")
    
    # Full-width sections
    if uploaded_file is not None and predicted_species is not None:
        # Confidence chart
        st.markdown("### 📊 Top 5 Predictions")
        confidence_chart = create_confidence_chart(top_predictions)
        st.plotly_chart(confidence_chart, use_container_width=True)
        
        # Detailed predictions table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Detailed Predictions")
            predictions_df = pd.DataFrame({
                'Rank': range(1, 6),
                'Species': [pred[0].title() for pred in top_predictions],
                'Confidence': [f"{pred[1]:.1%}" for pred in top_predictions],
                'Score': [pred[1] for pred in top_predictions]
            })
            
            # Style the dataframe
            styled_df = predictions_df.style.format({
                'Score': '{:.4f}'
            }).background_gradient(subset=['Score'], cmap='RdYlGn')
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Prediction Metrics")
            st.markdown(f"""
            <div class="model-metrics">
                <p><strong>🥇 Top Prediction:</strong> {top_predictions[0][0].title()}</p>
                <p><strong>📊 Confidence:</strong> {top_predictions[0][1]:.1%}</p>
                <p><strong>🎲 Uncertainty:</strong> {1-top_predictions[0][1]:.1%}</p>
                <p><strong>📈 Top-2 Gap:</strong> {(top_predictions[0][1] - top_predictions[1][1]):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Species information
        st.markdown("### 🌿 Species Information")
        display_tree_info(predicted_species)
        
        # Analysis timestamp
        st.markdown(f"""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <small>Analysis completed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌳 AI Tree Species Classifier | Built with Streamlit & TensorFlow</p>
        <p>🤖 Powered by Deep Learning | 📊 30 Tree Species Supported</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
