import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthcareChatbot:
    def __init__(self):
        """Initialize the healthcare chatbot with necessary components."""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize the language model
            self.chatbot = pipeline("text-generation", model="distilgpt2")
            
            # Load healthcare knowledge base
            self.knowledge_base = self.load_knowledge_base()
            
            # Initialize conversation history
            self.conversation_history = []
            
            # Load emergency keywords
            self.emergency_keywords = [
                "emergency", "severe", "critical", "heart attack", "stroke", 
                "unconscious", "bleeding", "suicide", "life-threatening"
            ]
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def load_knowledge_base(self) -> Dict:
        """Load healthcare-specific responses and information."""
        return {
            "symptoms": {
                "patterns": ["symptom", "feel", "pain", "ache", "hurt", "sore"],
                "responses": [
                    "Based on your description, I understand you're experiencing health concerns. "
                    "While I can provide general information, it's important to consult a healthcare "
                    "professional for proper diagnosis and treatment. Would you like information about "
                    "finding a doctor?",
                ],
                "follow_up_questions": [
                    "How long have you been experiencing these symptoms?",
                    "Have you consulted a healthcare provider about this?",
                    "Are the symptoms constant or do they come and go?"
                ]
            },
            "appointments": {
                "patterns": ["appointment", "schedule", "book", "visit", "see doctor"],
                "responses": [
                    "I can help you understand the appointment booking process. "
                    "Would you like information about:\n"
                    "1. Finding a specialist\n"
                    "2. Emergency care locations\n"
                    "3. Scheduling routine check-ups\n"
                    "4. Preparing for your appointment",
                ],
                "follow_up_questions": [
                    "Do you have a preferred healthcare provider?",
                    "Is this for a routine check-up or a specific concern?",
                    "What type of specialist are you looking for?"
                ]
            },
            "medications": {
                "patterns": ["medication", "medicine", "prescription", "drug", "pills"],
                "responses": [
                    "Medication management is crucial for your health. Remember to:\n"
                    "1. Take medications as prescribed\n"
                    "2. Keep a regular schedule\n"
                    "3. Store medicines properly\n"
                    "4. Contact your healthcare provider about side effects\n"
                    "5. Never share prescriptions with others",
                ],
                "follow_up_questions": [
                    "Do you need help setting up medication reminders?",
                    "Are you experiencing any side effects?",
                    "Would you like information about medication interactions?"
                ]
            },
            "lifestyle": {
                "patterns": ["diet", "exercise", "sleep", "stress", "healthy", "lifestyle"],
                "responses": [
                    "Making healthy lifestyle choices is important for your overall well-being. "
                    "Key areas to focus on include:\n"
                    "1. Balanced nutrition\n"
                    "2. Regular physical activity\n"
                    "3. Adequate sleep\n"
                    "4. Stress management\n"
                    "5. Regular health check-ups",
                ],
                "follow_up_questions": [
                    "Would you like specific information about any of these areas?",
                    "Have you set any health-related goals?",
                    "What aspects of your lifestyle would you like to improve?"
                ]
            }
        }

    def check_emergency(self, text: str) -> Tuple[bool, str]:
        """Check if the input contains emergency keywords."""
        emergency_found = any(keyword in text.lower() for keyword in self.emergency_keywords)
        emergency_message = (
            "⚠️ This sounds like an emergency situation. Please call emergency services "
            "(911 in the US) immediately or go to the nearest emergency room. "
            "Do not wait for an online response."
        ) if emergency_found else ""
        return emergency_found, emergency_message

    def preprocess_input(self, text: str) -> str:
        """Clean and normalize user input."""
        try:
            # Remove special characters and extra spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text.lower())
            processed_tokens = [word for word in tokens if word not in stop_words]
            
            # POS tagging to keep important words
            pos_tags = nltk.pos_tag(processed_tokens)
            important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ']
            processed_tokens = [word for word, pos in pos_tags if pos in important_pos]
            
            return " ".join(processed_tokens)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return text

    def get_follow_up_question(self, category: str) -> str:
        """Get a relevant follow-up question based on the conversation category."""
        if category in self.knowledge_base:
            questions = self.knowledge_base[category]["follow_up_questions"]
            # Get a question based on conversation history to avoid repetition
            used_questions = [msg.get("follow_up") for msg in self.conversation_history]
            available_questions = [q for q in questions if q not in used_questions]
            return available_questions[0] if available_questions else ""
        return ""

    def get_response(self, user_input: str) -> Dict[str, str]:
        """Generate a response based on user input."""
        try:
            # Check for emergency keywords first
            is_emergency, emergency_message = self.check_emergency(user_input)
            if is_emergency:
                return {
                    "response": emergency_message,
                    "follow_up": "",
                    "category": "emergency"
                }
            
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            
            # Check knowledge base for relevant responses
            for category, data in self.knowledge_base.items():
                if any(pattern in processed_input for pattern in data["patterns"]):
                    follow_up = self.get_follow_up_question(category)
                    return {
                        "response": data["responses"][0],
                        "follow_up": follow_up,
                        "category": category
                    }
            
            # If no specific healthcare response, use the language model
            response = self.chatbot(
                user_input,
                max_length=100,
                num_return_sequences=1,
                pad_token_id=50256,
                temperature=0.7
            )
            
            # Clean up the generated response
            generated_text = response[0]['generated_text']
            final_response = generated_text[len(user_input):].strip()
            
            if not final_response:
                final_response = (
                    "I apologize, but I'm not sure how to respond to that. "
                    "Could you please rephrase your question?"
                )
            
            return {
                "response": final_response,
                "follow_up": "",
                "category": "general"
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "follow_up": "",
                "category": "error"
            }

    def update_conversation_history(self, user_input: str, response_data: Dict[str, str]):
        """Store conversation history with metadata."""
        self.conversation_history.append({
            "user": user_input,
            "bot": response_data["response"],
            "category": response_data["category"],
            "follow_up": response_data.get("follow_up", ""),
            "timestamp": datetime.now().isoformat()
        })

def create_ui():
    """Create and configure the Streamlit UI."""
    st.set_page_config(
        page_title="Healthcare Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: white;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 0.8rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .bot-message {
            background-color: #f5f5f5;
            padding: 0.8rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .follow-up {
            color: #666;
            font-style: italic;
            margin-top: 0.5rem;
        }
        .emergency {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            border: 1px solid #ef9a9a;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    create_ui()
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HealthcareChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar with information
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        This healthcare assistant can help you with:
        - General health information
        - Symptom guidance
        - Appointment scheduling info
        - Medication reminders
        - Lifestyle advice
        
        Remember: This is not a substitute for professional medical care.
        """)
        
        # Add filters for conversation history
        if st.session_state.messages:
            st.markdown("### Filter Conversation")
            category_filter = st.multiselect(
                "Filter by category:",
                ["symptoms", "appointments", "medications", "lifestyle", "general"]
            )

    # Main chat interface
    st.title("🏥 Healthcare Assistant")
    st.markdown("*Your AI healthcare companion for general information and guidance*")
    
    # Disclaimer
    st.info(
        "⚠️ This chatbot provides general information only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek the "
        "advice of your physician or other qualified health provider."
    )

    # Chat interface
    user_input = st.text_input(
        "How can I assist you today?",
        key="user_input",
        placeholder="Type your health-related question here..."
    )

    if st.button("Send", key="send"):
        if user_input:
            with st.spinner("Processing your query..."):
                try:
                    # Get chatbot response
                    response_data = st.session_state.chatbot.get_response(user_input)
                    
                    # Update conversation history
                    st.session_state.chatbot.update_conversation_history(user_input, response_data)
                    st.session_state.messages.append({
                        "user": user_input,
                        "bot": response_data["response"],
                        "follow_up": response_data.get("follow_up", ""),
                        "category": response_data["category"]
                    })
                    
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    st.error("I apologize, but I encountered an error. Please try again.")

    # Display conversation history
    if st.session_state.messages:
        st.markdown("### Conversation History")
        
        # Apply filters if any are selected
        filtered_messages = st.session_state.messages
        if 'category_filter' in locals() and category_filter:
            filtered_messages = [
                msg for msg in st.session_state.messages
                if msg["category"] in category_filter
            ]
        
        for message in reversed(filtered_messages):
            # User message
            st.markdown(
                f'<div class="user-message">👤 **You:** {message["user"]}</div>',
                unsafe_allow_html=True
            )
            
            # Bot response
            response_class = "emergency" if message["category"] == "emergency" else "bot-message"
            st.markdown(
                f'<div class="{response_class}">🤖 **Assistant:** {message["bot"]}</div>',
                unsafe_allow_html=True
            )
            
            # Follow-up question if available
            if message.get("follow_up"):
                st.markdown(
                    f'<div class="follow-up">💭 {message["follow_up"]}</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown("---")

if __name__ == "__main__":
    main()