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
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
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
    """Calls Veo 3.1 via Vertex AI using the exact logic for the newest generation model."""
    try:
        from google import genai
        from google.genai import types
        import time
        
        if "gcp_service_account" not in st.secrets:
            return "Error: GCP Service Account JSON not found in secrets."
            
        creds_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        client = genai.Client(
            vertexai=True, 
            project=creds_info["project_id"], 
            location="us-central1",
            credentials=credentials
        )
        
        source = types.GenerateVideosSource(
            prompt=f"Educational animation of {metaphor_description} in a cinematic style, high quality, 3D render, smooth motion",
        )

        config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=8,
            person_generation="allow_all",
            generate_audio=False,
            resolution="720p",
        )

        # Start the video generation
        operation = client.models.generate_videos(
            model="veo-3.1-generate-001", source=source, config=config
        )

        # Poll for completion
        max_retries = 30
        retries = 0
        while not operation.done and retries < max_retries:
            time.sleep(10)
            operation = client.operations.get(operation)
            retries += 1
            
        if not operation.done:
            return "Video generation is taking longer than expected. The AI is still rendering!"

        # Check for errors in the operation itself
        if hasattr(operation, 'error') and operation.error:
            err = operation.error
            err_msg = err.get('message') if isinstance(err, dict) else getattr(err, 'message', str(err))
            return f"Veo 3.1 Error: {err_msg}"

        response = operation.result
        if response and hasattr(response, 'generated_videos') and response.generated_videos:
            gen_video = response.generated_videos[0]
            
            # Very aggressive URI/Bytes extraction with Magic Number check
            try:
                import base64
                
                if hasattr(gen_video, 'video') and hasattr(gen_video.video, 'video_bytes') and gen_video.video.video_bytes:
                    data = gen_video.video.video_bytes
                    
                    # If it's a string, decode it
                    if isinstance(data, str):
                        return base64.b64decode(data)
                    
                    # If it's bytes, check if it's already a valid MP4 (starts with ftyp)
                    # MP4 magic number usually starts at byte 4: 'ftyp'
                    if isinstance(data, (bytes, bytearray)):
                        if b'ftyp' in data[:12]:
                            return data # It's a real MP4
                        
                        # If no ftyp, it's likely base64-encoded bytes. Decode it!
                        try:
                            decoded = base64.b64decode(data)
                            if b'ftyp' in decoded[:12]:
                                return decoded
                        except:
                            pass
                    
                    return data
                
                if hasattr(gen_video, 'video') and hasattr(gen_video.video, 'uri') and gen_video.video.uri:
                    return str(gen_video.video.uri)
                
                if hasattr(gen_video, 'uri') and gen_video.uri:
                    return str(gen_video.uri)
            except:
                pass
            
            return f"Video generated but URI extraction failed. Data: {str(gen_video)[:200]}"
        
        return "Video generation completed, but the result list was empty."

    except Exception as e:
        import traceback
        return f"Error with Veo 3.1: {str(e)}\n{traceback.format_exc()[:200]}"
    
    return "Unknown error occurred in video agent."

def get_lens_mapping(technical_concept: str, selected_lens: str) -> Dict[str, str]:
    """
    Agent A - The Context Mapper (Orchestrator).
    Maps technical terms to metaphorical equivalents based on the lens.
    """
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
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
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
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
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
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

def generate_visual_image(prompt: str):
    """Calls Imagen 3 via Vertex AI to generate the visual representation."""
    try:
        from google import genai
        
        if "gcp_service_account" not in st.secrets:
            return None, "Error: GCP Service Account not found for Image Generation."
            
        creds_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        client = genai.Client(
            vertexai=True, 
            project=creds_info["project_id"], 
            location="us-central1",
            credentials=credentials
        )
        
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config={
                'number_of_images': 1,
                'aspect_ratio': '16:9'
            }
        )
        
        if response and response.generated_images:
            image = response.generated_images[0]
            data = None
            
            # 1. Extract potential data field
            if hasattr(image, 'image') and hasattr(image.image, 'image_bytes'):
                data = image.image.image_bytes
            elif hasattr(image, 'image_bytes'):
                data = image.image_bytes
            elif hasattr(image, 'gcs_uri'):
                return image.gcs_uri, None
                
            if not data:
                return None, f"Image found but no data field identified. Attributes: {dir(image)}"

            # 2. Handle BytesIO or similar stream objects
            if hasattr(data, 'getvalue'):
                data = data.getvalue()

            # 3. Handle Base64 (The most likely culprit for 'UnidentifiedImageError')
            import base64
            
            # If it's a string, it's definitely either Base64 or a URL
            if isinstance(data, str):
                if data.startswith(('http://', 'https://')):
                    return data, None
                try:
                    return base64.b64decode(data), None
                except Exception:
                    return data, None # Fallback to raw string

            # If it's bytes, it might still be Base64-encoded bytes
            # PNG starts with \x89PNG, JPEG starts with \xff\xd8
            if isinstance(data, (bytes, bytearray)):
                if data.startswith(b'\x89PNG') or data.startswith(b'\xff\xd8'):
                    return data, None # It's already raw binary
                
                # Try decoding as base64 just in case
                try:
                    decoded = base64.b64decode(data)
                    if decoded.startswith(b'\x89PNG') or decoded.startswith(b'\xff\xd8'):
                        return decoded, None
                except Exception:
                    pass
            
            return data, None

        return None, "Image generation failed – no images returned."

    except Exception as e:
        return None, f"Error with Imagen 3: {str(e)}"

def video_visualizer_agent(technical_concept: str, selected_lens: str, mapping: Dict[str, str]) -> str:
    """
    Agent E - The Video Visualizer.
    Generates a cinematic video prompt for Veo 3.1.
    """
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
    prompt = f"""
    You are a cinematic director and AI video prompt engineer.
    Create a 1-paragraph highly detailed video prompt for Veo 3.1 that visualizes the technical concept of "{technical_concept}" through the lens of "{selected_lens}".
    
    Use these mapped elements:
    {json.dumps(mapping, indent=2)}
    
    The video should be an educational 3D animation, cinematic lighting, ultra-high quality.
    Describe the movement, the camera angle, and how the technical concept is being demonstrated through the metaphor.
    Return ONLY the prompt string.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def visualizer_agent(technical_concept: str, selected_lens: str, mapping: Dict[str, str]) -> str:
    """
    Agent D - The Visualizer.
    Generates a prompt for Imagen 3 based on the metaphor mapping.
    """
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    
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
