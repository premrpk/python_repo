import json
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class PDFSearchEngine:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        self.similarity_threshold = 0.65

    def load_and_process_pdf(self, pdf_path: str) -> List[Document]:
        """Load and split PDF document into chunks with metadata"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split(self.text_splitter)

        # Enhance metadata with content type detection
        for page in pages:
            page.metadata['content_type'] = self._detect_content_type(page.page_content)
            if page.metadata['content_type'] == 'clause':
                page.metadata['clause_title'] = self._extract_clause_title(page.page_content)

        return pages

    def _detect_content_type(self, text: str) -> str:
        """Detect if text is table, clause, or regular content"""
        # Table detection
        if any(char in text for char in ['│', '┌', '┐', '└', '┘', '+']) or \
           sum(line.count('  ') > 3 for line in text.split('\n')) > 3:
            return 'table'

        # Clause detection
        if any(pattern in text[:50] for pattern in
              ['Article', 'Section', 'Clause', '§', 'CHAPTER']) or \
           (len(text.split('\n')[0]) < 50 and \
            any(text.split('\n')[0].startswith(p) for p in ['1.', 'A.', 'I.'])):
            return 'clause'

        return 'text'

    def _extract_clause_title(self, text: str) -> str:
        """Extract the title/heading of a legal clause"""
        first_line = text.split('\n')[0].strip()
        return first_line if len(first_line) < 100 else first_line[:100] + "..."

    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        return FAISS.from_documents(documents, self.embedding_model)

    def embedded_search(self, documents: List[Document], query: str) -> List[Dict[str, Any]]:
        """Perform traditional text-based search"""
        query_lower = query.lower()
        results = []

        for doc in documents:
            content_lower = doc.page_content.lower()
            if query_lower in content_lower:
                result = {
                    "page": doc.metadata['page'],
                    "content": doc.page_content,
                    "content_type": doc.metadata['content_type'],
                    "match_type": "exact",
                    "source": doc.metadata['source']
                }
                if doc.metadata['content_type'] == 'clause':
                    result["clause_title"] = doc.metadata.get('clause_title', '')
                results.append(result)

        return results

    def semantic_search(self, vector_store, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store"""
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
        results = []

        for doc, score in docs_and_scores:
            if score > self.similarity_threshold:
                result = {
                    "page": doc.metadata['page'],
                    "content": doc.page_content,
                    "content_type": doc.metadata['content_type'],
                    "match_type": "semantic",
                    "similarity_score": float(score),
                    "source": doc.metadata['source']
                }
                if doc.metadata['content_type'] == 'clause':
                    result["clause_title"] = doc.metadata.get('clause_title', '')
                results.append(result)

        return results

    def search_pdf(self, pdf_path: str, query: str) -> Dict[str, Any]:
        """Main search function combining both approaches"""
        documents = self.load_and_process_pdf(pdf_path)
        vector_store = self.create_vector_store(documents)

        embedded_results = self.embedded_search(documents, query)
        semantic_results = self.semantic_search(vector_store, query)

        # Combine and deduplicate results
        combined_results = embedded_results + semantic_results
        unique_results = []
        seen = set()

        for result in combined_results:
            # Use page number and content start as identifier
            identifier = (result['page'], result['content'][:50])
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)

        # Sort semantic results by score (highest first)
        unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        return {
            "user_input": query,
            "pdf_file": pdf_path,
            "matches": unique_results,
            "stats": {
                "total_pages": max(doc.metadata['page'] for doc in documents),
                "total_chunks": len(documents),
                "exact_matches": len(embedded_results),
                "semantic_matches": len(semantic_results),
                "unique_matches": len(unique_results)
            }
        }

def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pdf_search_langchain.py <pdf_path> <query>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    query = sys.argv[2]

    search_engine = PDFSearchEngine()
    results = search_engine.search_pdf(pdf_path, query)

    # Print pretty JSON output
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()




# ## Requirements Installation
# 
# First install the required packages:
# 
# ```bash
# pip install langchain pypdf faiss-cpu sentence-transformers
# ```
# 
# For GPU support (optional):
# 
# ```bash
# pip install faiss-gpu
# ```
# 
# ## How to Use
# 
# Run the program from command line:
# 
# ```bash
# python pdf_search_langchain.py path/to/document.pdf "search description"
# ```
# 
# ## Example Output Structure
# 
# ```json
# {
#   "user_input": "intellectual property rights",
#   "pdf_file": "law_document.pdf",
#   "matches": [
#     {
#       "page": 12,
#       "content": "Section 4.2 Intellectual Property\nAll intellectual property rights related to...",
#       "content_type": "clause",
#       "clause_title": "Section 4.2 Intellectual Property",
#       "match_type": "exact",
#       "source": "law_document.pdf"
#     },
#     {
#       "page": 15,
#       "content": "The parties agree that any IP rights developed during...",
#       "content_type": "text",
#       "match_type": "semantic",
#       "similarity_score": 0.82,
#       "source": "law_document.pdf"
#     },
#     {
#       "page": 42,
#       "content": "Table 3: IP Rights Transfer Conditions\n| Type | Duration | Conditions |\n|------|----------|------------|...",
#       "content_type": "table",
#       "match_type": "semantic",
#       "similarity_score": 0.71,
#       "source": "law_document.pdf"
#     }
#   ],
#   "stats": {
#     "total_pages": 50,
#     "total_chunks": 215,
#     "exact_matches": 1,
#     "semantic_matches": 4,
#     "unique_matches": 3
#   }
# }
# ```
# 
# ## Key Enhancements with LangChain:
# 
# 1. **Advanced Document Processing**:
#    - Uses LangChain's `PyPDFLoader` for robust PDF loading
#    - Implements `RecursiveCharacterTextSplitter` for optimal chunking
#    - Preserves metadata including page numbers and source
# 
# 2. **Improved Semantic Search**:
#    - Utilizes FAISS vector store for efficient similarity search
#    - Leverages HuggingFace embeddings for better semantic understanding
#    - Configurable similarity threshold and result count
# 
# 3. **Enhanced Content Detection**:
#    - Better table detection with multiple heuristics
#    - Improved clause identification with pattern matching
#    - Content type classification (text/table/clause)
# 
# 4. **Comprehensive Output**:
#    - Maintains source document information
#    - Includes content type classification
#    - Preserves clause titles when available
#    - Detailed statistics about the search results
# 
# 5. **Performance Considerations**:
#    - FAISS provides efficient vector similarity search
#    - Chunking balances context preservation with search precision
#    - Configurable parameters for different use cases
# 
# For production use, you might want to add:
# - Persistent vector store to avoid re-processing
# - Batch processing for multiple documents
# - API endpoint for web service integration
# - More sophisticated table extraction using dedicated libraries like Camelot or Tabula
