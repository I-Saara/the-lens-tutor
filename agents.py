import json
import google.generativeai as genai
from typing import Dict
import streamlit as st
from google.oauth2 import service_account

def configure_gemini(api_key: str):
    """Configure the Gemini API key and Vertex AI."""
    genai.configure(api_key=api_key)
    
    # Vertex AI initialization using Streamlit Secrets
    try:
        import vertexai
        if "gcp_service_account" in st.secrets:
            creds_info = st.secrets["gcp_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            vertexai.init(
                project=creds_info["project_id"],
                location="us-central1",
                credentials=credentials
            )
        else:
            # Fallback to default (for local testing with gcloud)
            vertexai.init(location="us-central1")
    except Exception as e:
        print(f"Vertex AI Init Note: {e}")

def generate_lesson_video(metaphor_description: str):
    """Calls the Veo model via Vertex AI to generate an educational animation."""
    try:
        import vertexai
        try:
            from vertexai.preview.vision_models import VideoGenerationModel
        except ImportError:
            # Fallback for newer versions where it might have moved out of preview
            from vertexai.vision_models import VideoGenerationModel
        
        model = VideoGenerationModel("veo-001")
        video_response = model.generate_video(
            prompt=f"Educational animation of {metaphor_description} in a cinematic style, high quality, 3D render",
            aspect_ratio="16:9",
            duration=5
        )
        # Note: In a real app, you'd handle the VideoGenerationResponse object
        # For this conceptual logic, we return the first video's path/url if available
        return video_response[0].video_uri if video_response else None
    except Exception as e:
        return f"Error generating video: {str(e)}"

def get_lens_mapping(technical_concept: str, selected_lens: str) -> Dict[str, str]:
    """
    Agent A - The Context Mapper (Orchestrator).
    Maps technical terms to metaphorical equivalents based on the lens.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    prompt = f"""
    You are an expert Context Mapper.
    Your job is to map the technical concept "{technical_concept}" to the lens of "{selected_lens}".
    Identify 3-5 key technical terms related to "{technical_concept}" and provide a metaphorical equivalent for each within the "{selected_lens}" context.
    
    Return the result strictly as a JSON object where keys are the technical terms and values are the metaphors.
    Do not include markdown blocks or any other text. Just the JSON.
    Example for Recursion in Chef lens:
    {{
      "Base Case": "The finished dish",
      "Recursive Step": "Adding a pinch of salt and tasting again",
      "Call Stack": "The stack of dirty plates"
    }}
    """
    
    # Using JSON generation config
    response = model.generate_content(
        prompt, 
        generation_config={"response_mime_type": "application/json"}
    )
    
    if not response.text:
        raise RuntimeError("Gemini returned an empty response – check model availability and prompt.")
    
    try:
        mapping = json.loads(response.text)
        return mapping
    except json.JSONDecodeError:
        # Fallback if parsing fails
        return {}

def teacher_agent(technical_concept: str, selected_lens: str, mapping: Dict[str, str]) -> str:
    """
    Agent B - The Teacher.
    Delivers the lesson using the generated metaphors.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    mapping_str = "\n".join([f'- {tech}: "{metaphor}"' for tech, metaphor in mapping.items()])
    
    prompt = f"""
    You are an expert tutor teaching the concept of "{technical_concept}" using ONLY the metaphor of "{selected_lens}".
    Here is your vocabulary mapping:
    {mapping_str}
    
    Explain the concept to a beginner. 
    IMPORTANT INSTRUCTION: Whenever you use one of the metaphorical terms from the mapping, you MUST wrap it in exactly this HTML tag: <span class="tooltip" title="TECHNICAL_TERM">METAPHOR</span>. 
    For example, if the mapping is "Base Case": "The finished dish", you write: <span class="tooltip" title="Base Case">The finished dish</span>.
    
    Do NOT use the technical terms directly in your explanation, only use the metaphors (wrapped in the span tag).
    Keep the explanation engaging, accurate to the metaphor, and around 2-3 paragraphs.
    """
    response = model.generate_content(prompt)
    if not response.text:
        raise RuntimeError("Teacher Agent returned an empty response.")
    return response.text

def fact_checker_agent(technical_concept: str, explanation: str) -> str:
    """
    Agent C - The Fact-Checker.
    Verifies the technical accuracy of the metaphorical explanation.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    prompt = f"""
    You are a Fact-Checker Agent.
    A teacher has explained the technical concept of "{technical_concept}" using a metaphor.
    Here is their explanation:
    {explanation}
    
    Your job is to verify if the technical essence of the concept is preserved despite the metaphor.
    Provide a brief (1 paragraph) assessment of the accuracy. Point out any flaws or confirm that the analogy holds up technically.
    """
    response = model.generate_content(prompt)
    if not response.text:
        raise RuntimeError("Fact-Checker Agent returned an empty response.")
    return response.text

def visualizer_agent(technical_concept: str, selected_lens: str, mapping: Dict[str, str]) -> str:
    """
    Agent D - The Visualizer.
    Generates a prompt for Imagen 3 based on the metaphor mapping.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    prompt = f"""
    You are an expert prompt engineer for Imagen 3.
    Create a highly detailed, visually striking image generation prompt that represents the technical concept of "{technical_concept}" entirely through the visual lens of "{selected_lens}".
    Use the following mapped elements in your visual description:
    {json.dumps(mapping, indent=2)}
    
    The prompt should specify lighting, mood, style, and composition. Return only the prompt string, no extra text.
    """
    response = model.generate_content(prompt)
    if not response.text:
        raise RuntimeError("Visualizer Agent returned an empty response.")
    return response.text
