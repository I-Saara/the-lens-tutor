import streamlit as st
from agents import configure_gemini, get_lens_mapping, teacher_agent, fact_checker_agent, visualizer_agent, generate_visual_image, generate_lesson_video, video_visualizer_agent

# Set page config
st.set_page_config(page_title="The Lens Adaptive Tutor", page_icon="🔍", layout="wide")

# Custom CSS for Vibe-Switch
VIBE_CSS = {
    "Trekker": """
        <style>
        .stApp { background-color: #e8f5e9; color: #1b5e20; }
        [data-testid="stHeader"] { background-color: transparent; }
        .stTextInput>div>div>input { background-color: #c8e6c9; color: #1b5e20; }
        .stSelectbox>div>div>div { background-color: #c8e6c9; color: #1b5e20; }
        h1, h2, h3, h4, p, span, label { color: #1b5e20 !important; }
        </style>
    """,
    "MasterChef": """
        <style>
        .stApp { background-color: #fff3e0; color: #e65100; }
        [data-testid="stHeader"] { background-color: transparent; }
        .stTextInput>div>div>input { background-color: #ffe0b2; color: #e65100; }
        .stSelectbox>div>div>div { background-color: #ffe0b2; color: #e65100; }
        h1, h2, h3, h4, p, span, label { color: #e65100 !important; }
        </style>
    """,
    "Founder": """
        <style>
        .stApp { background-color: #121212; color: #e0e0e0; }
        [data-testid="stHeader"] { background-color: transparent; }
        .stTextInput>div>div>input { background-color: #1e1e1e; color: #fff; }
        .stSelectbox>div>div>div { background-color: #1e1e1e; color: #fff; }
        h1, h2, h3, h4, p, span, label { color: #e0e0e0 !important; }
        .stExpander { background-color: #1e1e1e; }
        </style>
    """,
    "Pro-Gamer": """
        <style>
        .stApp { background-color: #1a0033; color: #e0b3ff; }
        [data-testid="stHeader"] { background-color: transparent; }
        .stTextInput>div>div>input { background-color: #2d004d; color: #fff; }
        .stSelectbox>div>div>div { background-color: #2d004d; color: #fff; }
        h1, h2, h3, h4, p, span, label { color: #e0b3ff !important; }
        .stExpander { background-color: #2d004d; }
        </style>
    """,
    "Surprise": """
        <style>
        .stApp { background-color: #fce4ec; color: #880e4f; }
        [data-testid="stHeader"] { background-color: transparent; }
        .stTextInput>div>div>input { background-color: #f8bbd0; color: #880e4f; }
        .stSelectbox>div>div>div { background-color: #f8bbd0; color: #880e4f; }
        h1, h2, h3, h4, p, span, label { color: #880e4f !important; }
        </style>
    """
}

# Custom CSS for the Translation Toggle (Tooltips)
TOOLTIP_CSS = """
<style>
.tooltip {
  position: relative;
  display: inline-block;
  border-bottom: 2px dashed currentColor;
  cursor: help;
  font-weight: bold;
}

.tooltip::after {
  content: attr(title);
  position: absolute;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%);
  background-color: rgba(0, 0, 0, 0.8);
  color: #fff !important;
  padding: 8px 12px;
  border-radius: 6px;
  white-space: nowrap;
  font-size: 14px;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.3s, visibility 0.3s;
  z-index: 1000;
  box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
}

.tooltip:hover::after {
  opacity: 1;
  visibility: visible;
}
</style>
"""
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio")

if api_key:
    st.session_state["api_key"] = api_key
    configure_gemini(api_key)
else:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to start learning.")
    st.stop()

# Main Application Title
st.title("🔍 The Lens Adaptive Tutor")
st.markdown("### *Learn any technical concept through the lens of your interests!*")

# Input Section
col1, col2 = st.columns(2)

with col1:
    concept = st.text_input("🧠 Technical Concept", placeholder="e.g., Recursion, React Hooks, DNS")

with col2:
    mode = st.selectbox("🎭 Select Lens", ["Trekker", "MasterChef", "Founder", "Pro-Gamer", "Surprise"])

custom_lens = None
if mode == "Surprise":
    custom_lens = st.text_input("✨ Enter your custom interest", placeholder="e.g., K-Pop, Cricket, Formula 1")

lens_to_use = custom_lens if mode == "Surprise" and custom_lens else mode

# Initialize session state for persistence
if "mapping" not in st.session_state:
    st.session_state.mapping = None
if "lesson" not in st.session_state:
    st.session_state.lesson = None
if "fact_check" not in st.session_state:
    st.session_state.fact_check = None
if "viz_prompt" not in st.session_state:
    st.session_state.viz_prompt = None
if "video_data" not in st.session_state:
    st.session_state.video_data = None
if "video_viz_prompt" not in st.session_state:
    st.session_state.video_viz_prompt = None

# Generate Action
if concept and lens_to_use:
    # Inject Vibe CSS
    vibe = mode if mode in VIBE_CSS else "Surprise"
    st.markdown(VIBE_CSS[vibe], unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🚀 Generate Lesson"):
        # Reset previous state
        st.session_state.mapping = None
        st.session_state.lesson = None
        st.session_state.fact_check = None
        st.session_state.viz_prompt = None
        st.session_state.video_data = None
        st.session_state.video_viz_prompt = None
            
        # 1. Map Context
        with st.spinner(f"Mapping '{concept}' to the world of '{lens_to_use}'..."):
            st.session_state.mapping = get_lens_mapping(concept, lens_to_use)
            
        if st.session_state.mapping:
            # 2. Teach
            with st.spinner("Preparing your lesson..."):
                st.session_state.lesson = teacher_agent(concept, lens_to_use, st.session_state.mapping)
                
            # 3. Fact Check
            with st.spinner("Fact-checker verifying technical accuracy..."):
                st.session_state.fact_check = fact_checker_agent(concept, st.session_state.lesson)
                
            # 4. Visualize
            with st.spinner("Drafting image prompt..."):
                st.session_state.viz_prompt = visualizer_agent(concept, lens_to_use, st.session_state.mapping)
                
            # 5. Video Prompt
            with st.spinner("Scripting video animation..."):
                st.session_state.video_viz_prompt = video_visualizer_agent(concept, lens_to_use, st.session_state.mapping)

    # Display Results (Outside of the button block so they persist)
    if st.session_state.mapping:
        with st.expander("🧩 View Context Mapping (Agent A)", expanded=False):
            st.json(st.session_state.mapping)
            
        if st.session_state.lesson:
            st.markdown("## 📖 The Lesson")
            st.info("💡 *Hover over the dashed terms to see their technical translation!*")
            with st.container():
                st.markdown(st.session_state.lesson, unsafe_allow_html=True)
            
        if st.session_state.fact_check:
            st.markdown("## ✅ Fact-Checker Assessment")
            st.success(st.session_state.fact_check)
            
        if st.session_state.viz_prompt:
            st.markdown("---")
            st.markdown("## 🎨 Visualizer (Imagen 3)")
            st.code(st.session_state.viz_prompt, language="text")
            
            if st.button("🖼️ Generate AI Visual"):
                with st.spinner("Bringing the metaphor to life with Imagen 3..."):
                    img_bytes, error = generate_visual_image(st.session_state.viz_prompt)
                    if img_bytes:
                        st.image(img_bytes, caption=f"{concept} through the lens of {lens_to_use}")
                        st.success("Visual generated successfully!")
                    else:
                        st.error(f"Failed to generate image: {error}")

            # 5. Video (Veo 3.1)
            st.markdown("---")
            st.markdown("## 🎬 Video Learning (Veo 3.1)")
            
            if st.button("🎥 Generate Video Animation"):
                with st.spinner("Creating cinematic 8s animation via Veo 3.1 (this may take up to 2 minutes)..."):
                    video_prompt = st.session_state.video_viz_prompt or st.session_state.mapping.get("metaphor_description", "The core concept")
                    video_url = generate_lesson_video(video_prompt)
                    if video_url and (not isinstance(video_url, str) or not video_url.startswith("Error")):
                        # If it's raw bytes, convert to a Data URI for persistence
                        if isinstance(video_url, bytes):
                            import base64
                            b64 = base64.b64encode(video_url).decode()
                            st.session_state.video_data = f"data:video/mp4;base64,{b64}"
                        else:
                            st.session_state.video_data = video_url
                    else:
                        st.error(f"Video generation failed: {video_url}")

            # Display the video if it exists in session state
            if "video_data" in st.session_state:
                video_val = st.session_state.video_data
                
                # If it's a Data URI or Bytes, try to make it a playable file
                if isinstance(video_val, (bytes, str)):
                    import tempfile
                    import os
                    
                    # Create a temp file to serve the video
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                        if isinstance(video_val, bytes):
                            tmpfile.write(video_val)
                        elif video_val.startswith("data:video/mp4;base64,"):
                            import base64
                            b64_data = video_val.split(",")[1]
                            tmpfile.write(base64.b64decode(b64_data))
                        else:
                            # If it's a URI string, just use it
                            tmpfile.write(video_val.encode() if isinstance(video_val, str) else video_val)
                            
                        temp_path = tmpfile.name
                    
                    st.video(temp_path)
                    st.success("Video generated successfully!")
                    
                    # Add a download button as a backup
                    with open(temp_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Video (.mp4)",
                            data=f,
                            file_name=f"tutor_animation_{concept}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.video(video_val)
