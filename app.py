import streamlit as st

def init_page_config():
    """Initialize the Streamlit page configuration."""
    st.set_page_config(
        page_title="MATLAB Troubleshooting AI | Tejas Ganesh",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Render the sidebar with project info and reset button."""
    with st.sidebar:
        st.title("⚙️ MATLAB AI")
        st.caption("Maintained by Tejas Ganesh")
        st.link_button("View project on GitHub", "https://github.com/tejasganesh04/matlab-troubleshooting-ai")
        st.markdown("---")
        st.markdown("### About Project")
        st.info(
            "This application is a professional MATLAB and Simulink Troubleshooting Assistant. "
            "It leverages advanced AI to analyze your questions and visual diagrams, providing "
            "step-by-step suggestions to help you investigate your issue."
        )
        st.markdown("---")
        if st.button("Reset Application", type="primary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

def main():
    """Main application layout and logic."""
    init_page_config()
    render_sidebar()

    # Apply custom CSS for clean, wide, professional look
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 2rem;
        }
        .response-container {
            background-color: #F3F4F6;
            padding: 2rem;
            border-radius: 8px;
            border-left: 4px solid #1E3A8A;
            margin-top: 2rem;
            color: #1F2937;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="main-header">MATLAB & Simulink Troubleshooting AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Describe your MATLAB/Simulink issue and optionally upload '
        'a relevant schematic, plot, or error screenshot to get an AI-generated solution.</div>',
        unsafe_allow_html=True
    )

    # Main content layout
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### Enter Your Question")
        user_question = st.text_area(
            "Question Input",
            placeholder="e.g., Why am I getting an algebraic loop error in my Simulink model?",
            height=250,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### Upload Image (Optional)")
        uploaded_image = st.file_uploader(
            "Image Input",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Image Preview", use_container_width=True)

    st.markdown("---")

    # Action buttons
    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    with btn_col1:
        generate_btn = st.button("Generate Answer", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if not generate_btn and st.session_state.get("last_answer"):
        st.markdown("### AI Troubleshooting Response")
        st.markdown(st.session_state["last_answer"])

    if st.session_state.get("last_answer") and not generate_btn:
        st.download_button("Download answer", st.session_state["last_answer"],
                           file_name="matlab-troubleshooting.md", mime="text/markdown")

    # Process generation
    if generate_btn:
        if not user_question.strip():
            st.error("Please enter a question to get a troubleshooting answer.", icon="🚨")
        else:
            with st.spinner("Analyzing query and generating response..."):
                try:
                    # Integrate with backend
                    from graph.builder import build_graph
                    
                    # Initialize and compile the LangGraph workflow
                    app = build_graph()
                    
                    # Read image bytes if uploaded
                    image_bytes = uploaded_image.getvalue() if uploaded_image else None
                    
                    # Prepare initial state for the graph
                    initial_state = {
                        "user_query": user_question,
                        "uploaded_image": image_bytes
                    }
                    
                    # Execute graph
                    result = app.invoke(initial_state)
                    
                    # Extract the generated answer
                    answer = result.get("generated_answer", "")
                    
                    if answer:
                        st.success("Troubleshooting analysis complete.", icon="✅")
                        st.markdown("### AI Troubleshooting Response")
                        st.session_state["last_answer"] = answer
                        st.markdown(answer)
                    else:
                        st.warning("The AI processed the request but returned no answer.", icon="⚠️")
                        
                except ImportError:
                    st.error(
                        "Backend modules not found. Please ensure the 'graph' package is available.",
                        icon="🚨"
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {str(e)}", icon="🚨")

if __name__ == "__main__":
    main()
