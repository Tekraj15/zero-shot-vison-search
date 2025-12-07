import os
import sys
import uuid
from tqdm import tqdm

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_loader import ModelLoader
from src.indexing_and_embedding import Indexer
from src.utils import get_image_paths, save_metadata, save_embeddings

def main():
    # Configuration
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets/image-dataset')
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    embeddings_path = os.path.join(data_dir, 'embeddings.npy')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    model_loader = ModelLoader()
    indexer = Indexer()
    
    # Get image paths
    print(f"Scanning {assets_dir} for images...")
    image_paths = get_image_paths(assets_dir)
    if not image_paths:
        print(f"No images found in {assets_dir}. Please add images.")
        return
    
    print(f"Found {len(image_paths)} images.")
    
    # Process images
    vectors = []
    metadata = {}
    all_embeddings = []
    
    print("Generating embeddings...")
    for img_path in tqdm(image_paths):
        # Generate unique ID for the image
        img_id = str(uuid.uuid4())
        
        # Get embedding
        embedding = model_loader.get_image_embedding(img_path)
        
        if embedding:
            # Store metadata
            # Calculate path relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            rel_path = os.path.relpath(img_path, project_root)
            meta = {
                "path": rel_path,
                "filename": os.path.basename(img_path)
            }
            
            vectors.append((img_id, embedding, meta))
            metadata[img_id] = meta
            all_embeddings.append(embedding)
            
    # Save local data
    print("Saving local data...")
    save_metadata(metadata, metadata_path)
    # save_embeddings(all_embeddings, embeddings_path) # Optional, can be large
    
    # Upsert to Pinecone
    if vectors:
        indexer.upsert_vectors(vectors)
        print("Ingestion complete!")
    else:
        print("No valid embeddings generated.")

if __name__ == "__main__":
    main()
