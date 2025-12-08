import streamlit as st
import os
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model_loader import ModelLoader
from src.vector_indexer import Indexer

# Page Config
st.set_page_config(
    page_title="Vision Scout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 1rem;
        border-radius: 10px;
    }
    .image-card {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .image-card:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    return ModelLoader(), Indexer()

def main():
    st.title("🔍 Vision Scout")
    st.markdown("### Zero-Shot Semantic Image Search")
    
    # Load components
    try:
        model_loader, indexer = load_components()
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.stop()
        
    # Search Bar
    query = st.text_input("Describe what you're looking for...", placeholder="e.g., 'a futuristic city at night' or 'a happy dog running'")
    
    if query:
        with st.spinner("Searching..."):
            # Generate text embedding
            text_embedding = model_loader.get_text_embedding(query)
            
            if text_embedding:
                # Search Pinecone
                results = indexer.search(text_embedding, top_k=12)
                
                if results and results['matches']:
                    st.markdown(f"Found **{len(results['matches'])}** matches for *'{query}'*")
                    
                    # Display results in a grid
                    cols = st.columns(3)
                    for idx, match in enumerate(results['matches']):
                        meta = match['metadata']
                        score = match['score']
                        
                        # Resolve image path
                        # Assuming assets are in the root assets/ folder relative to app.py
                        # The metadata path is relative like 'assets/image.jpg'
                        img_path = os.path.join(os.path.dirname(__file__), meta['path'])
                        
                        with cols[idx % 3]:
                            if os.path.exists(img_path):
                                image = Image.open(img_path)
                                st.image(image, use_container_width=True, caption=f"Score: {score:.2f}")
                            else:
                                st.warning(f"Image not found: {meta['path']}")
                else:
                    st.info("No matches found.")
            else:
                st.error("Failed to generate embedding for query.")

if __name__ == "__main__":
    main()
