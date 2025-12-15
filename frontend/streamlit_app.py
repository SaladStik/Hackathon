import io
import os
import requests
import streamlit as st

from PIL import Image



API_URL = os.getenv("API_URL", "http://localhost:8000")



#api functions

def detect_ppe(file):

    # detect ppe in the image provided
    try:
        files = {"file": (file.name, file.getvalue(), "image/jpeg")}
        response = requests.post(f"{API_URL}/detect", files=files, timeout=30)
        #st.write(response.json())
        if(response.status_code == 200):
            return response.json()
        return None
    except: 
        return None
def generate_report(file):
    # generate pdf report for the image provided
    try:
        files = {"file": (file.name, file.getvalue(), "image/jpeg")}
        response = requests.post(f"{API_URL}/report", files=files, timeout=30)
        if(response.status_code == 200):
            return response.content
        return None
    except: 
        return None
       
    



st.set_page_config(page_title="PPE Detector", page_icon="🦺", layout = "wide")


st.title("Personal Protective Equipment (PPE) Safety Detector")
st.divider()


#File uploader section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])



if uploaded_file:
    file_id = uploaded_file.file_id
    if st.session_state.get("last_file_id") != file_id:
        #loading
        with st.spinner("Analyzing image for PPE..."):
            
            uploaded_file.seek(0)
            result = detect_ppe(uploaded_file)
        
        if result:
            st.session_state["result"] = result
            st.session_state["last_file_id"] = file_id
            st.session_state["uploaded_file"] = uploaded_file

    #layout will be 2 columns: left for original image, right for result
    col_left, col_right = st.columns([1,1])
    with col_left:
        st.subheader("Original Image")
        #change image with to 400
        image = Image.open(uploaded_file)
        st.image(image, width=400)
    if "result" in st.session_state:
        result = st.session_state["result"]
        with col_right:
            #show result image
            if "annotated_image" in result:
                st.subheader("PPE Detection Result")
                
                # Display the annotated image
                import base64
                img_data = base64.b64decode(result["annotated_image"])
                st.image(img_data, width=400)

        # Compliance Summary - full width (outside columns)
        if "annotated_image" in result:
            total_people = result.get("total_persons", 0)
            compliance_summary = result.get("compliance_summary", {})
            compliant = compliance_summary.get("compliant", 0)
            
            # Count violations from summary object (items that are "MISSING")
            summary = result.get("summary", {})
            violations = sum(1 for status in summary.values() if status == "MISSING")

            st.subheader("Compliance Summary")
            col_metrics, col_chart = st.columns([2,1])

            with col_metrics:
                m1,m2,m3 = st.columns(3)
                m1.metric("Total Workers Detected", total_people)
                m2.metric("Compliant", compliant)
                m3.metric("Violations", violations)

            
            with col_chart:
                import plotly.express as px
                fig= px.pie(
                    names=["Compliant", "Violations"],
                    values=[compliant, violations],
                    #colours as hex
                    color_discrete_sequence=["#00cc96", "#ff6361"],
                    hole=0.4,
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    height= 150,
                    width=150,
                )
                fig.update_traces(textinfo='percent', textfont_size=14)
                st.plotly_chart(fig, use_container_width=True)
            
            #violations
            if violations > 0:
                st.error(f"Detected {violations} PPE Violations!")
                for person in result.get("persons", []):
                    status = person.get("compliance", "Unknown")
                    if status != "compliant":
                        with st.expander(f"Person ID: {person.get('person_id', 'N/A')} - Status: {status.upper()}"):
                            person_summary = person.get("summary", {})
                            #grab their ppe
                            for ppe_item, ppe_status in person_summary.items():
                                if ppe_status != "DETECTED":
                                    #if detected then show checkmark else cross
                                    symbol = "✅" if ppe_status == "DETECTED" else "❌"
                                    st.write(f"- {ppe_item}: {ppe_status} {symbol}")
            else:
                st.success("All workers are compliant with PPE requirements!")

            st.divider()
            
            #generate pdf report
            st.subheader("Generate Report")

            #if pdf data is not in session state
            if "pdf_data" not in st.session_state:
                st.session_state["pdf_data"] = None
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF Report..."):
                    st.session_state["uploaded_file"].seek(0)
                    pdf = generate_report(st.session_state["uploaded_file"])
                    if pdf:
                        st.session_state["pdf_data"] = pdf
                    else:
                        st.error("Failed to generate PDF report. Please try again.")
            if st.session_state.pdf_data:
                st.success("PDF Report Generated!")
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name="ppe_report.pdf",
                    mime="application/pdf",
                )