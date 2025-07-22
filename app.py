import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from dotenv import load_dotenv
import time
import re
from google.api_core.exceptions import ResourceExhausted
from langchain.schema import Document

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class BookSummarizer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,  # Lower temperature for more focused summaries
            request_timeout=120
        )
        self.vector_store = None
        self.book_text = ""
        self.chapters = []

    def extract_pdf_text(self, pdf_docs):
        """Extract text from uploaded PDF files"""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return text

    def detect_chapters(self, text):
        """Detect chapters in the book using common patterns"""
        chapter_patterns = [
            r'(?i)chapter\s+\d+',
            r'(?i)chapter\s+[ivxlcdm]+',  # Roman numerals
            r'(?i)part\s+\d+',
            r'(?i)section\s+\d+',
            r'\n\d+\.\s+[A-Z]',  # Numbered sections
            r'\n[A-Z][A-Z\s]+\n',  # All caps titles
        ]
        
        chapters = []
        for pattern in chapter_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start_pos = match.start()
                # Find the end of the line to get full chapter title
                line_end = text.find('\n', start_pos + len(match.group()))
                if line_end == -1:
                    line_end = start_pos + 100  # Fallback
                
                chapter_title = text[start_pos:line_end].strip()
                chapters.append({
                    'title': chapter_title,
                    'start_pos': start_pos,
                    'pattern': pattern
                })
        
        # Sort by position and remove duplicates
        chapters = sorted(chapters, key=lambda x: x['start_pos'])
        unique_chapters = []
        last_pos = 0
        for chapter in chapters:
            if chapter['start_pos'] > last_pos + 50:  # Avoid very close matches
                unique_chapters.append(chapter)
                last_pos = chapter['start_pos']
        
        return unique_chapters[:20]  # Limit to first 20 chapters

    def split_into_chapters(self, text, chapters):
        """Split book text into individual chapters"""
        if not chapters:
            return [{"title": "Full Book", "content": text}]
        
        chapter_contents = []
        for i, chapter in enumerate(chapters):
            start_pos = chapter['start_pos']
            end_pos = chapters[i + 1]['start_pos'] if i + 1 < len(chapters) else len(text)
            
            content = text[start_pos:end_pos].strip()
            if len(content) > 100:  # Only include substantial chapters
                chapter_contents.append({
                    "title": chapter['title'],
                    "content": content
                })
        
        return chapter_contents

    def create_text_chunks(self, text, chunk_size=8000, chunk_overlap=800):
        """Create text chunks for processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def create_vector_store(self, text_chunks):
        """Create FAISS vector store from text chunks"""
        try:
            documents = [Document(page_content=chunk) for chunk in text_chunks]
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local("book_faiss_index")
            return True
        except ResourceExhausted:
            st.error("Rate limit exceeded during vector store creation. Please wait and try again.")
            return False
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = FAISS.load_local(
                "book_faiss_index", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False

    def summarize_full_book(self, summary_length="medium"):
        """Generate summary of the entire book"""
        try:
            # Create prompts based on summary length
            if summary_length == "short":
                map_template = """
                Provide a concise summary (2-3 sentences) of the following text:
                {docs}
                Summary:
                """
                reduce_template = """
                Combine these summaries into a brief overview (1-2 paragraphs) of the entire book:
                {doc_summaries}
                
                Final Summary:
                """
            elif summary_length == "long":
                map_template = """
                Provide a detailed summary (1-2 paragraphs) of the following text, including key points and important details:
                {docs}
                Summary:
                """
                reduce_template = """
                Combine these summaries into a comprehensive overview (3-5 paragraphs) of the entire book, including:
                - Main themes and concepts
                - Key arguments or plot points
                - Important conclusions or takeaways
                
                Summaries to combine:
                {doc_summaries}
                
                Final Comprehensive Summary:
                """
            else:  # medium
                map_template = """
                Provide a balanced summary (1 paragraph) of the following text, covering main points:
                {docs}
                Summary:
                """
                reduce_template = """
                Combine these summaries into a well-structured overview (2-3 paragraphs) of the entire book:
                {doc_summaries}
                
                Final Summary:
                """

            map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])
            reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["doc_summaries"])

            # Create chains
            map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
            reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

            # Combine documents chain
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain,
                document_variable_name="doc_summaries"
            )

            # Map reduce chain
            reduce_documents_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_documents_chain,
                collapse_documents_chain=combine_documents_chain,
                token_max=4000,
            )

            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                reduce_documents_chain=reduce_documents_chain,
                document_variable_name="docs",
                return_intermediate_steps=False,
            )

            # Prepare documents
            text_chunks = self.create_text_chunks(self.book_text, chunk_size=6000)
            docs = [Document(page_content=chunk) for chunk in text_chunks[:15]]  # Limit for rate limits

            # Generate summary
            result = map_reduce_chain.run(docs)
            return result

        except ResourceExhausted:
            return "Rate limit exceeded. Please wait before generating summary."
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def summarize_chapter(self, chapter_title, summary_length="medium"):
        """Summarize a specific chapter"""
        try:
            # Find the chapter content
            chapter_content = None
            for chapter in self.chapters:
                if chapter_title.lower() in chapter['title'].lower() or chapter['title'].lower() in chapter_title.lower():
                    chapter_content = chapter['content']
                    break
            
            if not chapter_content:
                return f"Chapter '{chapter_title}' not found. Available chapters: {[ch['title'] for ch in self.chapters[:5]]}"

            # Create summary prompt based on length
            if summary_length == "short":
                prompt_template = """
                Provide a brief summary (2-3 sentences) of this chapter:
                
                Chapter Content:
                {text}
                
                Brief Summary:
                """
            elif summary_length == "long":
                prompt_template = """
                Provide a detailed summary (2-3 paragraphs) of this chapter, including:
                - Main topics discussed
                - Key points and arguments
                - Important examples or evidence
                - Conclusions reached
                
                Chapter Content:
                {text}
                
                Detailed Summary:
                """
            else:  # medium
                prompt_template = """
                Provide a balanced summary (1-2 paragraphs) of this chapter, covering the main points and key takeaways:
                
                Chapter Content:
                {text}
                
                Summary:
                """

            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Truncate if too long
            if len(chapter_content) > 15000:
                chapter_content = chapter_content[:15000] + "..."
            
            summary = chain.run(text=chapter_content)
            return summary

        except ResourceExhausted:
            return "Rate limit exceeded. Please wait before generating chapter summary."
        except Exception as e:
            return f"Error generating chapter summary: {str(e)}"

    def search_topic(self, topic_query, num_results=5):
        """Search for specific topic in the book"""
        try:
            if not self.vector_store:
                return "Vector store not available. Please process the book first."
            
            docs = self.vector_store.similarity_search(topic_query, k=num_results)
            
            prompt_template = """
            Based on the following excerpts from the book, provide a comprehensive summary about: {topic}
            
            Book excerpts:
            {context}
            
            Topic Summary:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["topic", "context"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            context = "\n\n".join([doc.page_content for doc in docs])
            summary = chain.run(topic=topic_query, context=context)
            
            return summary

        except ResourceExhausted:
            return "Rate limit exceeded. Please wait before searching topics."
        except Exception as e:
            return f"Error searching topic: {str(e)}"

def main():
    st.set_page_config(page_title="AI Book Summarizer", page_icon="📚", layout="wide")
    st.title("📚 AI Book Summarizer")
    st.markdown("Upload a PDF book and get summaries of the entire book, specific chapters, or search for topics!")

    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = BookSummarizer()
    
    if 'book_processed' not in st.session_state:
        st.session_state.book_processed = False

    # Sidebar for book upload and processing
    with st.sidebar:
        st.header("📖 Book Upload")
        uploaded_file = st.file_uploader("Upload PDF Book", type="pdf")
        
        if uploaded_file and st.button("Process Book"):
            with st.spinner("Processing book... This may take a few minutes."):
                try:
                    # Extract text
                    text = st.session_state.summarizer.extract_pdf_text([uploaded_file])
                    st.session_state.summarizer.book_text = text
                    
                    if len(text.strip()) == 0:
                        st.error("No text found in the PDF!")
                        return
                    
                    # Detect chapters
                    detected_chapters = st.session_state.summarizer.detect_chapters(text)
                    chapter_contents = st.session_state.summarizer.split_into_chapters(text, detected_chapters)
                    st.session_state.summarizer.chapters = chapter_contents
                    
                    # Create vector store
                    text_chunks = st.session_state.summarizer.create_text_chunks(text)
                    success = st.session_state.summarizer.create_vector_store(text_chunks)
                    
                    if success:
                        st.session_state.book_processed = True
                        st.success(f"Book processed successfully!")
                        st.info(f"📊 Stats:\n- {len(text):,} characters\n- {len(text_chunks)} chunks\n- {len(chapter_contents)} chapters detected")
                    else:
                        st.error("Failed to process book due to rate limits.")
                        
                except Exception as e:
                    st.error(f"Error processing book: {str(e)}")
        
        # Rate limit info
        st.markdown("---")
        st.markdown("### ⚠️ Rate Limit Info")
        st.markdown("- Using Gemini Free Tier")
        st.markdown("- Wait between requests")
        st.markdown("- Large books may hit limits")

    # Main interface
    if st.session_state.book_processed:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📋 Summary Options")
            
            summary_type = st.selectbox(
                "Choose Summary Type:",
                ["Full Book Summary", "Chapter Summary", "Topic Search"]
            )
            
            summary_length = st.selectbox(
                "Summary Length:",
                ["short", "medium", "long"],
                index=1
            )
            
            if summary_type == "Full Book Summary":
                if st.button("Generate Full Book Summary", type="primary"):
                    with st.spinner("Generating full book summary..."):
                        summary = st.session_state.summarizer.summarize_full_book(summary_length)
                        st.markdown("### 📖 Full Book Summary")
                        st.markdown(summary)
            
            elif summary_type == "Chapter Summary":
                available_chapters = [ch['title'] for ch in st.session_state.summarizer.chapters]
                if available_chapters:
                    selected_chapter = st.selectbox("Select Chapter:", available_chapters)
                    
                    if st.button("Generate Chapter Summary", type="primary"):
                        with st.spinner(f"Generating summary for: {selected_chapter}"):
                            summary = st.session_state.summarizer.summarize_chapter(selected_chapter, summary_length)
                            st.markdown(f"### 📑 Chapter Summary: {selected_chapter}")
                            st.markdown(summary)
                else:
                    st.warning("No chapters detected. Try Full Book Summary instead.")
            
            elif summary_type == "Topic Search":
                topic_query = st.text_input("Enter topic to search for:", placeholder="e.g., machine learning, character development, economic theory")
                
                if topic_query and st.button("Search Topic", type="primary"):
                    with st.spinner(f"Searching for: {topic_query}"):
                        summary = st.session_state.summarizer.search_topic(topic_query)
                        st.markdown(f"### 🔍 Topic Summary: {topic_query}")
                        st.markdown(summary)
        
        with col2:
            st.header("📑 Book Structure")
            
            if st.session_state.summarizer.chapters:
                st.markdown("**Detected Chapters:**")
                for i, chapter in enumerate(st.session_state.summarizer.chapters[:10], 1):
                    st.markdown(f"{i}. {chapter['title']}")
                
                if len(st.session_state.summarizer.chapters) > 10:
                    st.markdown(f"... and {len(st.session_state.summarizer.chapters) - 10} more")
            else:
                st.info("No clear chapter structure detected. Use Full Book Summary or Topic Search.")
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.markdown("- **Full Book**: Overview of entire content")
            st.markdown("- **Chapter**: Focused on specific sections")
            st.markdown("- **Topic Search**: Find specific themes")
            st.markdown("- **Length**: Short (few sentences) to Long (detailed)")

    else:
        st.info("👆 Please upload and process a PDF book to get started!")
        
        # Example usage
        st.markdown("### 🚀 How to Use")
        st.markdown("1. **Upload PDF**: Choose your book file")
        st.markdown("2. **Process**: Wait for text extraction and analysis")
        st.markdown("3. **Summarize**: Choose from multiple summary options")
        
        st.markdown("### ✨ Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📖 Full Book Summaries**\n- Complete overview\n- Adjustable length\n- Key themes extraction")
        with col2:
            st.markdown("**📑 Chapter Summaries**\n- Individual chapters\n- Auto-detection\n- Focused content")
        with col3:
            st.markdown("**🔍 Topic Search**\n- Find specific themes\n- Semantic search\n- Contextual summaries")

if __name__ == "__main__":
    main()