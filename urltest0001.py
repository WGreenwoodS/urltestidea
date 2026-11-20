import streamlit as st
import pandas as pd
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ------------------------------------------------------
# 1. Pydantic Schema for Strict JSON Output
# ------------------------------------------------------
class URLAnalysis(BaseModel):
    url: str = Field(description="The analyzed URL")
    clarity_score: int = Field(description="Score 1-10 on Above-the-Fold Clarity")
    social_proof_score: int = Field(description="Score 1-10 on Social Proof Density")
    friction_score: int = Field(description="Score 1-10 on Pricing & CTA Friction (10 = no friction, 1 = high friction)")
    justification: str = Field(description="1-2 brief sentences justifying the scores based on page content")

class MatrixResult(BaseModel):
    analyses: list[URLAnalysis]

# ------------------------------------------------------
# 2. Streamlit UI Setup
# ------------------------------------------------------
st.set_page_config(page_title="Competitor CRO Matrix", layout="wide")
st.title("📊 Competitor URL & Conversion Teardown Matrix")
st.markdown("""
**Marketing Portfolio Project:** This tool uses Gemini's native web grounding to read live competitor landing pages and grade their Conversion Rate Optimization (CRO) fundamentals. 
Instead of qualitative guesses, it generates a strict, benchmarked 1-10 scoring matrix.
""")

# Sidebar for API Key (test safely)
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get yours at aistudio.google.com")
    st.markdown("---")
    st.markdown("**Methodology & Human Audit:**\n* **Clarity:** 5-second rule visibility.\n* **Social Proof:** Named logos vs generic claims.\n* **Friction:** Transparent pricing vs 'Book a Demo' walls.\n* *Temperature set to 0.1 for consistent data.*")

# Input for URLs
st.subheader("Target URLs to Analyze")
col1, col2, col3 = st.columns(3)
with col1:
    url1 = st.text_input("Competitor 1 URL", value="https://stripe.com")
with col2:
    url2 = st.text_input("Competitor 2 URL", value="https://www.paypal.com")
with col3:
    url3 = st.text_input("Competitor 3 URL", value="https://squareup.com")

# ------------------------------------------------------
# 3. Execution & API Call
# ------------------------------------------------------
if st.button("Run Quantitative CRO Audit", type="primary"):
    urls_to_analyze = [u for u in [url1, url2, url3] if u.strip()]
    
    if not api_key:
        st.error("Please enter a Gemini API Key in the sidebar to run the audit.")
        st.stop()
        
    if len(urls_to_analyze) < 1:
        st.error("Please enter at least one URL.")
        st.stop()

    with st.spinner("Initializing Gemini API and analyzing live web context..."):
        try:
            client = genai.Client(api_key=api_key)
            
            prompt = f"""
            Act as an expert Growth Marketing and CRO auditor. 
            Analyze the following live landing pages: {', '.join(urls_to_analyze)}
            
            Evaluate them against this specific rubric on a 1 to 10 scale:
            1. Above-the-Fold Clarity: Can you tell exactly what they sell and who it is for within the first few headlines?
            2. Social Proof Density: Do they use hard numbers, real named logos, and concrete case studies (high score) or just generic claims (low score)?
            3. Pricing & CTA Friction: Do they have transparent pricing and easy CTAs (10) or do they hide behind gated 'Book a Demo' walls (1)?
            
            Return the output strictly adhering to the requested JSON schema.
            """
            
            # Call Gemini using Google Search Grounding to ensure it can access the live URLs
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1, # Low temperature for analytical consistency
                    response_mime_type="application/json",
                    response_schema=MatrixResult,
                    tools=[{"google_search": {}}] # Enables live web context
                )
            )
            
            # ------------------------------------------------------
            # 4. Data Processing & Visualization
            # ------------------------------------------------------
            result_dict = json.loads(response.text)
            df = pd.DataFrame(result_dict["analyses"])
            
            # Calculate an average "Conversion Readiness Score"
            df["conversion_readiness_avg"] = df[["clarity_score", "social_proof_score", "friction_score"]].mean(axis=1).round(1)
            
            st.success("Audit Complete!")
            
            # Layout the visual outputs
            chart_col, table_col = st.columns([1, 2])
            
            with chart_col:
                st.write("### Benchmark Comparison")
                # Prepare data for Streamlit's native bar chart
                chart_data = df.set_index("url")[["clarity_score", "social_proof_score", "friction_score"]]
                st.bar_chart(chart_data)
                
            with table_col:
                st.write("### CRO Scorecard Matrix")
                # Format dataframe for clean display
                display_df = df.set_index("url")
                st.dataframe(
                    display_df,
                    column_config={
                        "clarity_score": st.column_config.NumberColumn("Clarity", format="%d/10"),
                        "social_proof_score": st.column_config.NumberColumn("Social Proof", format="%d/10"),
                        "friction_score": st.column_config.NumberColumn("Friction (Low=Bad)", format="%d/10"),
                        "conversion_readiness_avg": st.column_config.NumberColumn("Total Readiness", format="%.1f/10"),
                        "justification": "Auditor Justification"
                    },
                    use_container_width=True
                )
            
            # Download button for recruiters to export the data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audit CSV",
                data=csv,
                file_name='competitor_cro_audit.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"An error occurred during the API call: {e}")