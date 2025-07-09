import os
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import BallTree
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import tempfile
from typing import List, Dict, Tuple, Union

class PDFVectorProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the PDF vector processor with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.metadata = {}
        self.tree = None
        self.vector_data = []
        self.text_chunks = []
        self.page_map = []
        
    def _extract_text_and_metadata(self, pdf_path: str) -> Tuple[List[str], Dict]:
        """
        Extract text and metadata from PDF with page numbers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (text_chunks, metadata)
        """
        text_chunks = []
        page_map = []
        metadata = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract document metadata
            doc_info = pdf_reader.metadata or {}
            metadata = {
                'title': doc_info.get('/Title', ''),
                'author': doc_info.get('/Author', ''),
                'creator': doc_info.get('/Creator', ''),
                'producer': doc_info.get('/Producer', ''),
                'subject': doc_info.get('/Subject', ''),
                'num_pages': len(pdf_reader.pages)
            }
            
            # Extract text from each page with page numbers
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text:
                    # Split text into chunks (you can adjust chunk size)
                    chunks = self._chunk_text(text, chunk_size=500)
                    for chunk in chunks:
                        text_chunks.append(chunk)
                        page_map.append(page_num)
        
        return text_chunks, page_map, metadata
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        
        Args:
            text: Text to chunk
            chunk_size: Approximate size of each chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(word)
            current_length += len(word) + 1
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def _extract_images(self, pdf_path: str, output_dir: str = None) -> List[Dict]:
        """
        Extract images from PDF and save them to output directory.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images (if None, uses temp directory)
            
        Returns:
            List of image metadata dictionaries
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            
        os.makedirs(output_dir, exist_ok=True)
        image_metadata = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page_{page_num+1}_img_{img_index}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                image_metadata.append({
                    'page': page_num + 1,
                    'path': image_path,
                    'width': base_image.get('width', 0),
                    'height': base_image.get('height', 0)
                })
        
        return image_metadata
    
    def pdf_to_vectors(self, pdf_path: str, output_file: str = None) -> Dict:
        """
        Convert PDF to vector embeddings and save to file.
        
        Args:
            pdf_path: Path to the PDF file
            output_file: File to save vector data (if None, returns dict without saving)
            
        Returns:
            Dictionary containing vector data and metadata
        """
        # Extract text and metadata
        self.text_chunks, self.page_map, self.metadata = self._extract_text_and_metadata(pdf_path)
        
        # Extract images
        image_metadata = self._extract_images(pdf_path)
        self.metadata['images'] = image_metadata
        
        # Generate embeddings
        embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)
        self.vector_data = embeddings.tolist()
        
        # Build search index
        self.tree = BallTree(embeddings, metric='euclidean')
        
        # Prepare output data
        output_data = {
            'metadata': self.metadata,
            'vectors': self.vector_data,
            'text_chunks': self.text_chunks,
            'page_map': self.page_map,
            'model': self.model.get_sentence_embedding_dimension()
        }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
        
        return output_data
    
    def load_vector_file(self, vector_file: str):
        """
        Load vector data from file.
        
        Args:
            vector_file: Path to the vector file
        """
        with open(vector_file, 'r') as f:
            data = json.load(f)
            
        self.metadata = data['metadata']
        self.vector_data = data['vectors']
        self.text_chunks = data['text_chunks']
        self.page_map = data['page_map']
        
        # Rebuild search index
        self.tree = BallTree(np.array(self.vector_data), metric='euclidean')
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform semantic search on the PDF content.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with text, page number, and score
        """
        query_embedding = self.model.encode([query])
        distances, indices = self.tree.query(query_embedding, k=k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'text': self.text_chunks[idx],
                'page': self.page_map[idx],
                'score': float(distances[0][i])
            })
        
        return results
    
    def vectors_to_pdf(self, output_pdf: str):
        """
        Reconstruct a PDF from vector data (approximate reconstruction).
        
        Args:
            output_pdf: Path to save the reconstructed PDF
        """
        doc = fitz.open()
        
        current_page_num = 1
        page = doc.new_page()
        
        # Add metadata
        if 'title' in self.metadata:
            doc.set_metadata({
                'title': self.metadata['title'],
                'author': self.metadata.get('author', ''),
                'subject': self.metadata.get('subject', '')
            })
        
        # Reconstruct text content
        font_size = 11
        margin = 50
        y_position = margin
        
        for i, (text, page_num) in enumerate(zip(self.text_chunks, self.page_map)):
            # Create new page if needed
            if page_num != current_page_num:
                page = doc.new_page()
                current_page_num = page_num
                y_position = margin
            
            # Insert text
            rc = page.insert_text(
                point=fitz.Point(margin, y_position),
                text=text,
                fontsize=font_size,
                fontname="helv",
                color=(0, 0, 0)
            
            y_position += font_size * 1.5 * (text.count('\n') + 1)
            
            # Check if we need a new page due to vertical space
            if y_position > page.rect.height - margin:
                page = doc.new_page()
                current_page_num += 1
                y_position = margin
        
        # Reinsert images if available
        if 'images' in self.metadata:
            for img_info in self.metadata['images']:
                page_num = img_info['page']
                
                # Make sure we have enough pages
                while len(doc) < page_num:
                    doc.new_page()
                
                page = doc[page_num - 1]
                
                try:
                    img = fitz.Pixmap(img_info['path'])
                    rect = fitz.Rect(
                        margin,
                        margin,
                        min(margin + img_info['width'], page.rect.width - margin),
                        min(margin + img_info['height'], page.rect.height - margin)
                    )
                    page.insert_image(rect, pixmap=img)
                except Exception as e:
                    print(f"Could not insert image {img_info['path']}: {str(e)}")
        
        # Save the PDF
        doc.save(output_pdf)
        doc.close()


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PDFVectorProcessor()
    
    # Convert PDF to vectors and save to file
    pdf_path = "example.pdf"
    vector_file = "example_vectors.json"
    processor.pdf_to_vectors(pdf_path, vector_file)
    
    # Perform semantic search
    query = "important concept"
    results = processor.semantic_search(query)
    print("Search results:")
    for result in results:
        print(f"Page {result['page']} (Score: {result['score']:.3f}): {result['text'][:100]}...")
    
    # Reconstruct PDF from vectors
    reconstructed_pdf = "reconstructed.pdf"
    processor.vectors_to_pdf(reconstructed_pdf)
    print(f"Reconstructed PDF saved to {reconstructed_pdf}")
    
    # Alternative: Load from existing vector file
    new_processor = PDFVectorProcessor()
    new_processor.load_vector_file(vector_file)
    new_results = new_processor.semantic_search("another query")
    
    # pip install pypdf2 sentence-transformers scikit-learn pymupdf pillow numpy