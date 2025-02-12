import streamlit as st
from transformers import pipeline

# Load a medical Question Answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Healthcare knowledge base
medical_knowledge = {
    "cold and cough": """
        Common medications for cold and cough include:
        - **Paracetamol** for fever and pain relief.
        - **Antihistamines** (e.g., cetirizine) for runny nose.
        - **Decongestants** (e.g., pseudoephedrine) for nasal congestion.
        - **Cough suppressants** (e.g., dextromethorphan) for dry cough.
        - **Expectorants** (e.g., guaifenesin) to loosen mucus.
        - **Home remedies** include warm fluids, honey, and steam inhalation.
        - **Consult a doctor** if symptoms persist for more than a week.
    """,
    "fever": """
        - **Paracetamol** (acetaminophen) is commonly used to reduce fever.
        - Stay **hydrated** and rest to help recovery.
        - Seek medical help if fever is **above 102°F (39°C)** or lasts more than 3 days.
    """,
    "appointment": "Would you like assistance in booking a doctor's appointment?",
    "hospital": "Please enter your location (latitude, longitude) to find nearby hospitals."
}

# Function to handle chatbot responses
def healthcare_chatbot(user_input):
    user_input = user_input.lower()

    # Check for predefined medical responses
    for key in medical_knowledge:
        if key in user_input:
            return medical_knowledge[key]

    # Use the AI model for general medical Q&A
    context = " ".join(medical_knowledge.values())  # Combine all knowledge as context
    response = qa_pipeline(question=user_input, context=context)
    
    return response["answer"] if response["answer"] else "I'm here to assist with general healthcare questions. Please consult a professional for medical advice."

# Streamlit web app
def main():
    st.title("Healthcare Assistant Chatbot")

    # Input box for user queries
    user_input = st.text_input("How can I assist you today?", "")

    # Button to process input
    if st.button("Submit"):
        if user_input.strip():
            st.write("**User:**", user_input)

            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)

            st.write("**Healthcare Assistant:**", response)

        else:
            st.warning("Please enter a query.")  # Display warning for empty input

if __name__ == "__main__":
    main()
