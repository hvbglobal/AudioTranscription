import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel
from fpdf import FPDF
from datetime import datetime
import platform
import re
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
from transformers import AutoTokenizer
GROQ_API_KEY = "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"


# Set page configuration
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="wide"
)
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')

tokenizer = load_tokenizer()

# Create a title for the app
st.title("🎙️ Audio Transcription with Whisper AI")
st.markdown("Upload an audio file (MP3 or WAV) and get it transcribed to text.")

# Model selection
model_size = st.sidebar.selectbox(
    "Select Whisper Model Size",
    ["tiny", "base", "small", "medium", "large-v2"]
)

# Define compute type options based on platform
compute_options = ["auto", "cpu"]
if platform.system() != "Darwin" or not platform.machine().startswith("arm"):  # Not Apple Silicon
    compute_options.insert(1, "cuda")  # Add CUDA as an option for non-Apple Silicon

compute_type = st.sidebar.radio(
    "Select Compute Device",
    compute_options
)

# Define precision options
precision_type = st.sidebar.radio(
    "Select Compute Precision",
    ["auto", "float32", "float16", "int8"]
)

st.sidebar.info(
    "Model sizes from small to large offer increasing accuracy but require more processing time and resources. "
    "'tiny' and 'base' are fastest but less accurate. 'small' offers a good balance. "
    "'medium' and 'large-v2' are more accurate but slower."
)

# Language selection (optional)
language = st.sidebar.selectbox(
    "Select Language (Optional)",
    ["Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese"]
)

language_code_map = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja"
}

# Upload file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

# LLM API Configuration
with st.sidebar.expander("LLM Settings"):
    use_llm = st.checkbox("Use Groq LLM for enhanced transcript processing", value=True)
    llm_model = st.selectbox(
        "Select LLM Model",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"]
    )



# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = None
if 'enhanced_transcription' not in st.session_state:
    st.session_state.enhanced_transcription = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Function to generate PDF
def generate_pdf(text, filename, title="Transcription"):
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"{title} of {filename}", ln=True, align="C")
    pdf.ln(10)
    
    # Add timestamp
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Add content
    pdf.set_font("Arial", size=12)
    
    # Split text into lines to avoid overflow
    pdf.multi_cell(0, 10, text)
    
    # Generate a filename for the PDF
    pdf_filename = f"{title.lower()}_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Save the PDF
    pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
    pdf.output(pdf_path)
    
    return pdf_path, pdf_filename

# Function to process audio and get transcription
def transcribe_audio(audio_file, model_size, compute_type, precision_type, language_code):
    try:
        with st.spinner(f"Loading Whisper model ({model_size})..."):
            # Initialize the model with proper error handling for compute type
            try:
                # If auto is selected for precision, let the library decide
                compute_precision = precision_type if precision_type != "auto" else None
                
                model = WhisperModel(model_size, device=compute_type, compute_type=compute_precision)
                st.info(f"Model loaded successfully using {compute_type} device with {model.compute_type} precision.")
            except Exception as e:
                st.warning(f"Failed to initialize with selected options: {str(e)}. Falling back to CPU with float32.")
                model = WhisperModel(model_size, device="cpu", compute_type="float32")
        
        with st.spinner("Transcribing audio... This may take a while depending on the file size and model."):
            # Save upload to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name
            
            # Transcribe the audio
            transcribe_options = {}
            if language_code:
                transcribe_options["language"] = language_code
            
            segments, info = model.transcribe(tmp_path, beam_size=5, **transcribe_options)
            
            # Combine all segments into one text
            transcript = ""
            for segment in segments:
                transcript += segment.text + " "
            
            # Clean up the temporary file
            os.unlink(tmp_path)
            
            return transcript.strip()
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return None

# Function to enhance transcript using Groq API
def enhance_transcript_with_llm(transcript,model):
    if not GROQ_API_KEY:
        st.warning("No Groq API Key found. Cannot enhance transcript.")
        return None

    
    try:
        with st.spinner("Enhancing transcript with LLM..."):
            headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

            
            prompt = f"""You are an expert transcriptionist. Below is a raw transcript from an audio file. 
            Please improve it by:
            1. Fixing grammar and punctuation errors
            2. Identifying and properly formatting paragraphs
            3. Identifying speakers if there are multiple (mark them as Speaker 1, Speaker 2, etc.)
            4. Removing filler words and stammers
            5. Do not change the content or meaning of the text

            Here is the raw transcript:
            {transcript}
            
            Enhanced transcript:"""
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4096
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                    headers=headers, 
                                    data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                enhanced_text = result["choices"][0]["message"]["content"]
                return enhanced_text
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        st.error(f"Error enhancing transcript: {str(e)}")
        return None

# Function to setup and get embedding model
def get_embedding_model():
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model for RAG..."):
            try:
                # Use a smaller model for embeddings to improve speed
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.embedding_model = model
                return model
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return None
    return st.session_state.embedding_model

# Function to chunk text for embeddings
def chunk_text(text, chunk_size=100, overlap=20):
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        
        if current_length >= chunk_size:
            chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
            
            # Handle overlap
            if overlap > 0:
                current_chunk = current_chunk[-overlap:]
                current_length = len(current_chunk)
            else:
                current_chunk = []
                current_length = 0
    
    if current_chunk:
        chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

# Function to generate questions using RAG
def generate_questions_rag(transcript, api_key, llm_model, num_questions=50):
    if not transcript or len(transcript.strip()) < 100:
        return ["The transcript is too short to generate meaningful questions."]
    
    try:
        # Load embedding model
        model = get_embedding_model()
        if model is None:
            return generate_questions_fallback(transcript, num_questions)
        
        # Chunk the transcript
        chunks = chunk_text(transcript)
        if not chunks:
            return generate_questions_fallback(transcript, num_questions)
        
        # Generate embeddings for chunks
        with st.spinner("Generating embeddings for RAG..."):
            chunk_embeddings = model.encode(chunks)
            
            # Create FAISS index
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(chunk_embeddings).astype('float32'))
        
        # Generate diverse question patterns
        question_patterns = [
            "What is the main topic discussed in the section about {topic}?",
            "How does the speaker describe {topic}?",
            "Why is {topic} significant according to the transcript?",
            "What evidence supports the claims about {topic}?",
            "What are the implications of {topic} mentioned in the transcript?",
            "How would you compare the discussion about {topic} with other perspectives?",
            "What conclusions can be drawn from the section about {topic}?",
            "What questions remain unanswered about {topic}?",
            "How does {topic} relate to the broader themes in the transcript?",
            "What might be some counterarguments to the points made about {topic}?"
        ]
        
        # If we have API key, use LLM for better questions
        if api_key:
            with st.spinner("Generating questions with LLM based on RAG..."):
                # Sample diverse chunks to ensure topic coverage
                selected_chunks = []
                if len(chunks) <= 5:
                    selected_chunks = chunks
                else:
                    step = max(1, len(chunks) // 5)
                    for i in range(0, len(chunks), step):
                        if len(selected_chunks) < 5:
                            selected_chunks.append(chunks[i])
                
                # Build prompt with selected chunks
            context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(selected_chunks)])
                
            prompt = f"""You are an experienced A level and IGCSE exam writer specialized in creating listening comprehension questions.

                
                EXAM SPECIFICATION: A levels or IGCSE
                
                QUESTION TYPES TO INCLUDE: Multiple Choice Questions
                
                Below are key segments from a transcript of an audio recording. Create exactly {num_questions} 
                high-quality listening exam questions based on this content.
                
                Each question should:
                1. Be clearly numbered
                2. Follow standard IGCSE question formats
                3. Test comprehension appropriate to IGCSE standards
                4. Include clear instructions where needed (e.g., "Choose ONE answer", "Complete the sentence", etc.)
                5. For multiple choice questions, include all options labeled A, B, C, (and D if four options)
                6. Include mark allocations when appropriate (e.g., [1 mark], [2 marks])
                
                TRANSCRIPT SEGMENTS:
                {context}
                
                Generate exactly {num_questions} exam-quality questions suitable for an A level and IGCSE listening paper.
                Format them as they would appear in an actual exam paper with question numbers and appropriate spacing.
                Do not include answers or mark schemes."""
                
            headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
            payload = {
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
                
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                        headers=headers, 
                                        data=json.dumps(payload))
                
            if response.status_code == 200:
                    result = response.json()
                    questions_text = result["choices"][0]["message"]["content"]
                    
                    # Extract questions from the response
                    questions = []
                    for line in questions_text.strip().split("\n"):
                        # Match patterns like "1.", "1)", "Question 1:", etc.
                        if re.match(r"^\d+[\.\):]", line.strip()):
                            # Remove the number and any special characters at the start
                            cleaned_question = re.sub(r"^\d+[\.\):\s]+", "", line.strip())
                            questions.append(cleaned_question)
                        elif line.strip() and not line.strip().startswith("Question") and not line.strip().startswith("#"):
                            # Catch any other lines that might be questions without numbering
                            questions.append(line.strip())
                    
                    if questions and len(questions) >= 5:
                        return questions[:num_questions]
        
        # If LLM fails or no API key, fall back to RAG-based generation
        questions = []
        for i in range(min(num_questions, len(chunks))):
            chunk = chunks[i]
            
            # Extract a topic from the chunk
            words = chunk.split()
            if len(words) < 5:
                continue
            
            # Choose a key phrase from the chunk (simplified)
            start_idx = min(3, len(words) - 3)  # Skip first few words which might be connecting words
            phrase_length = min(3, len(words) - start_idx)
            topic = " ".join(words[start_idx:start_idx + phrase_length])
            
            # Select a question pattern
            pattern_idx = i % len(question_patterns)
            question = question_patterns[pattern_idx].format(topic=topic)
            questions.append(question)
        
        # If we still don't have enough questions, add fallback questions
        if len(questions) < num_questions:
            fallback = generate_questions_fallback(transcript, num_questions - len(questions))
            questions.extend(fallback)
        
        return questions[:num_questions]
            
    except Exception as e:
        st.error(f"Error in RAG question generation: {str(e)}")
        return generate_questions_fallback(transcript, num_questions)

# Fallback question generation without RAG
def generate_questions_fallback(transcript, num_questions=50):
    try:
        tokens = tokenizer.tokenize(transcript)
        if len(tokens) < 20:
            return ["The transcript doesn't contain enough content to generate meaningful questions."]
        
        # Split into small chunks
        chunks = chunk_text(transcript, chunk_size=100, overlap=20)
        
        if not chunks:
            return ["Failed to chunk transcript for fallback questions."]
        
        max_questions = min(num_questions, len(chunks))
        
        selected_chunks = []
        step = max(1, len(chunks) // max_questions)
        for i in range(0, len(chunks), step):
            if len(selected_chunks) < max_questions:
                selected_chunks.append(chunks[i])
            else:
                break
        
        questions = []
        for chunk in selected_chunks:
            clean_chunk = chunk.replace('\n', ' ').strip()
            words = clean_chunk.split()
            if len(words) >= 5:
                question = f"What does the transcript say about {' '.join(words[:5])}...?"
                questions.append(question)
            
            if len(questions) >= max_questions:
                break
        
        # Add general questions if needed
        general_questions = [
            "What is the main topic discussed in this transcript?",
            "Who are the main speakers or subjects mentioned in the transcript?",
            "What are the key points made in this conversation?",
            "Can you summarize the main argument or narrative in the transcript?",
            "What important details are mentioned in the transcript?",
            "What conclusions can be drawn from the information in the transcript?",
            "How would you describe the tone or style of this transcript?",
            "What questions remain unanswered based on this transcript?",
            "What evidence or examples are provided to support the main ideas?",
            "How does this transcript relate to broader contexts or issues?"
        ]
        
        while len(questions) < max_questions and general_questions:
            questions.append(general_questions.pop(0))
        
        # Ensure no duplicates
        unique_questions = list(dict.fromkeys(questions))
        return unique_questions[:max_questions]
    
    except Exception as e:
        st.error(f"An error occurred while generating fallback questions: {str(e)}")
        return ["Error generating questions. Please try again."]


# Transcribe button
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.audio(uploaded_file, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
    
    with col2:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
            "File type": uploaded_file.type
        }
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
    
    if st.button("Transcribe Audio"):
        language_code = language_code_map[language]
        transcription = transcribe_audio(uploaded_file, model_size, compute_type, precision_type, language_code)
        
        if transcription:
            st.session_state.transcription = transcription
            st.session_state.filename = uploaded_file.name
            st.session_state.generated_questions = None
            st.session_state.enhanced_transcription = None
            st.success("Transcription complete!")
            
            # If LLM is enabled and API key is provided, enhance the transcript
            if use_llm and GROQ_API_KEY:
                enhanced = enhance_transcript_with_llm(transcription, llm_model)
                if enhanced:
                    st.session_state.enhanced_transcription = enhanced
                    st.success("Transcript enhanced with LLM!")

# Display transcription if available
if st.session_state.transcription:
    # If enhanced version is available, show tabs
    if st.session_state.enhanced_transcription:
        tab1, tab2 = st.tabs(["Enhanced Transcript", "Original Transcript"])
        
        with tab1:
            st.subheader("Enhanced Transcript")
            st.text_area("LLM-Enhanced Text", st.session_state.enhanced_transcription, height=300)
            
            # Download options for enhanced transcript
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Enhanced Text",
                    data=st.session_state.enhanced_transcription,
                    file_name=f"enhanced_{os.path.splitext(st.session_state.filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            with col2:
                if st.button("Save Enhanced as PDF"):
                    with st.spinner("Generating PDF..."):
                        pdf_path, pdf_filename = generate_pdf(st.session_state.enhanced_transcription, 
                                                             st.session_state.filename, 
                                                             "Enhanced Transcription")
                        with open(pdf_path, "rb") as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="Download Enhanced PDF",
                            data=pdf_data,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                        os.unlink(pdf_path)
        
        with tab2:
            st.subheader("Original Transcript")
            st.text_area("Transcribed Text", st.session_state.transcription, height=300)
    else:
        # Just show the original transcript
        st.subheader("Transcription Result")
        st.text_area("Transcribed Text", st.session_state.transcription, height=300)
    
    # Options section for transcript
    st.subheader("Transcript Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Save as Text File"):
            text_to_save = st.session_state.enhanced_transcription if st.session_state.enhanced_transcription else st.session_state.transcription
            prefix = "enhanced_" if st.session_state.enhanced_transcription else ""
            st.download_button(
                label="Download Text",
                data=text_to_save,
                file_name=f"{prefix}transcription_{os.path.splitext(st.session_state.filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("Save as PDF"):
            with st.spinner("Generating PDF..."):
                text_to_save = st.session_state.enhanced_transcription if st.session_state.enhanced_transcription else st.session_state.transcription
                title = "Enhanced Transcription" if st.session_state.enhanced_transcription else "Transcription"
                pdf_path, pdf_filename = generate_pdf(text_to_save, st.session_state.filename, title)
                
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
                
                os.unlink(pdf_path)
    
    with col3:
        if st.button("Generate Questions (RAG)"):
            with st.spinner("Generating questions using RAG..."):
                text_to_use = st.session_state.enhanced_transcription if st.session_state.enhanced_transcription else st.session_state.transcription
                st.session_state.generated_questions = generate_questions_rag(text_to_use, GROQ_API_KEY, llm_model, 50)
                st.success("Questions generated!")

# Display generated questions if available
if st.session_state.generated_questions:
    st.subheader("Generated Questions")
    
    for i, question in enumerate(st.session_state.generated_questions, 1):
        st.markdown(f"{i}. {question}")
    
    # Options to save questions
    col1, col2 = st.columns(2)
    
    with col1:
        questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(st.session_state.generated_questions, 1)])
        st.download_button(
            label="Download Questions as Text",
            data=questions_text,
            file_name=f"questions_{os.path.splitext(st.session_state.filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        if st.button("Save Questions as PDF"):
            with st.spinner("Generating Questions PDF..."):
                questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(st.session_state.generated_questions, 1)])
                pdf_path, pdf_filename = generate_pdf(questions_text, st.session_state.filename, "Questions")
                
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                st.download_button(
                    label="Download Questions PDF",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
                
                os.unlink(pdf_path)

# Requirements info
with st.expander("Additional Requirements"):
    st.markdown("""
    This app requires the following Python packages:
    ```
    streamlit
    faster-whisper
    fpdf
    nltk
    requests
    sentence-transformers
    faiss-cpu (or faiss-gpu)
    ```
    
    Install with:
    ```
    pip install streamlit faster-whisper fpdf nltk requests sentence-transformers faiss-cpu
    ```
    """)

# Footer
st.markdown("---")
st.markdown("### How to use this app")
st.markdown("""
1. Upload an audio file (MP3, WAV, or M4A format)
2. Select the Whisper model size (tiny is fastest, large-v2 is most accurate)
3. Choose your compute device (CPU or GPU if available)
4. Select compute precision (float32 is most compatible)
5. Optionally select a language (if known) to improve transcription
6. Configure Groq API settings in the sidebar for enhanced transcription
7. Click 'Transcribe Audio' to process the file
8. Once transcription is complete, you can:
   - View both original and enhanced transcript (if LLM enhancement was used)
   - Save transcripts as text files or PDFs
   - Generate questions using RAG (Retrieval-Augmented Generation)
   - Save the generated questions as text or PDF
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app uses Faster Whisper for audio transcription, Groq API for transcript enhancement, "
    "and Retrieval-Augmented Generation (RAG) to create meaningful questions. "
    "The transcription quality depends on the model size and audio quality. "
    "For better results with non-English content, try selecting the specific language."
)