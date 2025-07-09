import json
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import re


class PDFSearchEngine:
    def __init__(self):
        # Initialize the semantic model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = 0.6  # Similarity threshold for semantic search

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers and metadata"""
        extracted_data = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()

                # Basic table detection (look for patterns common in tables)
                is_table = any(char in text for char in ['│', '┌', '┐', '└', '┘', '+']) or \
                          any(line.count('\t') > 2 or line.count('  ') > 3 for line in text.split('\n'))

                extracted_data.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'is_table': is_table,
                    'clauses': self._identify_clauses(text)
                })

        return extracted_data

    def _identify_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Identify law clauses in text"""
        clauses = []
        lines = text.split('\n')
        current_clause = None

        for line in lines:
            line = line.strip()
            # Simple pattern matching for clause numbers (e.g., "1.1", "Article 3", "Section 5.2")
            if any(pattern in line for pattern in ['Article', 'Section', 'Clause', '§']) or \
               (len(line) < 20 and re.match(r'^\d+\.\d+', line)):
                if current_clause:
                    clauses.append(current_clause)
                current_clause = {'title': line, 'content': ''}
            elif current_clause:
                current_clause['content'] += line + '\n'

        if current_clause:
            clauses.append(current_clause)

        return clauses

    def embedded_search(self, data: List[Dict[str, Any]], description: str) -> List[Dict[str, Any]]:
        """Perform text-based (embedded) search"""
        results = []
        description_lower = description.lower()

        for item in data:
            # Search in regular text
            if description_lower in item['text'].lower():
                results.append({
                    'type': 'text',
                    'page': item['page_number'],
                    'content': item['text'],
                    'match_type': 'exact'
                })

            # Search in tables
            if item['is_table'] and description_lower in item['text'].lower():
                results.append({
                    'type': 'table',
                    'page': item['page_number'],
                    'content': item['text'],
                    'match_type': 'exact'
                })

            # Search in clauses
            for clause in item['clauses']:
                if description_lower in clause['title'].lower() or \
                   description_lower in clause['content'].lower():
                    results.append({
                        'type': 'clause',
                        'page': item['page_number'],
                        'title': clause['title'],
                        'content': clause['content'],
                        'match_type': 'exact'
                    })

        return results

    def semantic_search(self, data: List[Dict[str, Any]], description: str) -> List[Dict[str, Any]]:
        """Perform semantic (meaning-based) search"""
        results = []
        desc_embedding = self.model.encode([description])

        for item in data:
            # Encode the entire page text
            page_embedding = self.model.encode([item['text']])
            similarity = cosine_similarity(desc_embedding, page_embedding)[0][0]

            if similarity > self.threshold:
                results.append({
                    'type': 'text',
                    'page': item['page_number'],
                    'content': item['text'],
                    'similarity_score': float(similarity),
                    'match_type': 'semantic'
                })

            # Check individual clauses
            for clause in item['clauses']:
                clause_text = clause['title'] + ' ' + clause['content']
                clause_embedding = self.model.encode([clause_text])
                similarity = cosine_similarity(desc_embedding, clause_embedding)[0][0]

                if similarity > self.threshold:
                    results.append({
                        'type': 'clause',
                        'page': item['page_number'],
                        'title': clause['title'],
                        'content': clause['content'],
                        'similarity_score': float(similarity),
                        'match_type': 'semantic'
                    })

        # Sort by similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return results

    def search_pdf(self, pdf_path: str, description: str) -> Dict[str, Any]:
        """Main search function combining both approaches"""
        extracted_data = self.extract_text_from_pdf(pdf_path)

        embedded_results = self.embedded_search(extracted_data, description)
        semantic_results = self.semantic_search(extracted_data, description)

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

        return {
            'user_input': description,
            'pdf_file': pdf_path,
            'matches': unique_results,
            'stats': {
                'total_pages': len(extracted_data),
                'embedded_matches': len(embedded_results),
                'semantic_matches': len(semantic_results),
                'unique_matches': len(unique_results)
            }
        }

def main():
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pdf_search.py <pdf_path> <description>")
        return

    pdf_path = sys.argv[1]
    description = sys.argv[2]
    pdf_path = "F:\\desk_downloads\\git_src\\python-basics\\core\\src\\ARS_701.0.pdf"

    search_engine = PDFSearchEngine()
    results = search_engine.search_pdf(pdf_path, description)

    # Print pretty JSON output
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()



# 
# ## Requirements Installation
# 
# First install the required packages:
# 
# ```bash
# pip install PyPDF2 sentence-transformers scikit-learn
# pip install tf-keras
# ```
# 
# ## How to Use
# 
# Run the program from command line:
# 
# ```bash
# python pdf_search.py path/to/document.pdf "search description"
# ```
#   python pdf_search.py ARS_701.0.pdf.pdf "Owner-occupied (housing loan)"
# ## Example Output Structure
# 
# ```json
# {
#   "user_input": "intellectual property rights",
#   "pdf_file": "law_document.pdf",
#   "matches": [
#     {
#       "type": "clause",
#       "page": 12,
#       "title": "Section 4.2 Intellectual Property",
#       "content": "All intellectual property rights related to...",
#       "match_type": "exact"
#     },
#     {
#       "type": "text",
#       "page": 15,
#       "content": "The parties agree that any IP rights developed...",
#       "similarity_score": 0.78,
#       "match_type": "semantic"
#     }
#   ],
#   "stats": {
#     "total_pages": 42,
#     "embedded_matches": 1,
#     "semantic_matches": 3,
#     "unique_matches": 2
#   }
# }
# ```
# 
# ## Key Features:
# 
# 1. **Dual Search Approach**:
#    - Embedded search for exact text matches
#    - Semantic search for conceptual matches using sentence embeddings
# 
# 2. **Structured Content Identification**:
#    - Detects tables based on common patterns
#    - Extracts law clauses using pattern matching
# 
# 3. **Comprehensive Output**:
#    - Preserves original formatting where possible
#    - Includes page numbers for reference
#    - Provides match type and confidence scores
# 
# 4. **Readable JSON Format**:
#    - Properly indented output
#    - Clear structure with user input and statistics
# 
# Note: For very large PDFs, you might want to add pagination or limits to the output. The semantic search uses the 'all-MiniLM-L6-v2' model which provides a good balance between accuracy and performance.
# 