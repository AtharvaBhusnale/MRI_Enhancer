import json
import google.generativeai as genai

class Chatbot:
    def __init__(self, api_key, knowledge_base_path):
        # 1. Configure Gemini
        genai.configure(api_key=api_key)
        
        # Configure safety settings to prevent blocking on medical context
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        self.model = genai.GenerativeModel('gemini-flash-latest', safety_settings=safety_settings)
        
        # 2. Load Knowledge Base
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # 3. Initialize Chat Session with System Context
        self.system_context = self.construct_system_context()
        self.chat = self.model.start_chat(history=[])
        
        # Pre-seed the chat with the system context to enforce RAG behavior
        try:
            self.chat.send_message(self.system_context)
        except Exception as e:
            print(f"⚠️ Warning: Could not send system context to Gemini (likely Quota Exceeded). Chatbot started in fallback mode. Error: {e}")

    def load_knowledge_base(self, filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Knowledge base file '{filepath}' not found.")
            return {}

    def construct_system_context(self):
        """Creates a prompt that enforces using ONLY the knowledge base."""
        kb_str = json.dumps(self.knowledge_base, indent=2)
        return f"""
        You are an AI Medical Assistant. Your goal is to assess patient symptoms comfortably and professionally, using the provided Knowledge Base as your clinical guide.
        
        --- KNOWLEDGE BASE START ---
        {kb_str}
        --- KNOWLEDGE BASE END ---
        
        INSTRUCTIONS:
        1.  **Be Natural & Empathetic**: Speak like a caring medical professional. Do NOT mention "severity rules", "scores", "knowledge base", or "JSON" to the patient.
        2.  **Assessment**: Use the questions in the Knowledge Base to guide the conversation. If the user mentions a symptom, ask the relevant clarifying questions naturally.
        3.  **No Diagnosis**: Never give a medical diagnosis. Instead, assess urgency.
        4.  **Internal Scoring**:
            *   Evaluate the user's input against the "severity_rules" in your head.
            *   At the very end of your response, you MUST append a HIDDEN JSON block with the estimated severity score.
            *   Format: ```json{{"score": <number>, "symptom": "<symptom_name>"}}```
            *   If you are just asking a question and haven't determined severity yet, output score: 0.
        
        Example Interaction:
        User: "My head hurts."
        You: "I'm sorry to hear that. Could you tell me a bit more? On a scale of 1 to 10, how severe is the pain?" ```json{{"score": 0, "symptom": "headache"}}```
        User: "It's about a 3."
        You: "Okay, a mild headache can still be uncomfortable. Have you experienced any dizziness with it?" ```json{{"score": 0, "symptom": "headache"}}```
        """

    def process_message(self, user_input):
        """
        Sends user input to Gemini and parses the response.
        Returns: (reply_string, score_to_add, matched_symptom)
        """
        try:
            response = self.chat.send_message(user_input)
            text_response = response.text
            
            # Parse out the hidden JSON score
            score = 0
            symptom = None
            clean_reply = text_response
            
            if "```json" in text_response:
                try:
                    # Extract JSON block
                    json_block = text_response.split("```json")[1].split("```")[0]
                    data = json.loads(json_block)
                    score = data.get("score", 0)
                    symptom = data.get("symptom", None)
                    
                    # Remove JSON from the reply shown to user
                    clean_reply = text_response.split("```json")[0].strip()
                except Exception as e:
                    print(f"Error parsing bot JSON: {e}")
            
            return clean_reply, score, symptom
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Print full stack trace to console
            print(f"Gemini API Error Detail: {e}")
            return "I'm having trouble connecting to my medical database right now. Please try again.", 0, None

    def generate_summary(self, final_score, symptoms_list):
        """Generates a final summary based on the total severity score."""
        # We can also use the LLM for this, but the rule-based logic is fine for now.
        summary = "--- Final Summary ---\n"
        if not symptoms_list:
            summary += "We didn't get to discuss any specific symptoms."
            return summary

        summary += f"Symptoms discussed: {', '.join(symptoms_list)}\n"
        summary += f"Final Severity Score: {final_score}\n\n"

        # Use KB recommendations
        recs = self.knowledge_base.get('recommendations', {})
        if final_score >= 10:
            summary += f"Recommendation: {recs.get('high', 'Seek help.')}"
        elif final_score >= 5:
            summary += f"Recommendation: {recs.get('medium', 'See a doctor.')}"
        else:
            summary += f"Recommendation: {recs.get('low', 'Monitor symptoms.')}"

        return summary
