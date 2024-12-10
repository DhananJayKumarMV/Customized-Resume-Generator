# Step 1: Import Necessary Libraries
from langchain.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import PyPDF2
from fpdf import FPDF
 
# Step 2: Initialize the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key='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',  # Replace with your actual API key
    model_name="llama-3.1-70b-versatile"
)
 
# Step 3: Extract Job Requirements
def extract_job_requirements(job_url):
    loader = UnstructuredURLLoader(urls=[job_url])
    docs = loader.load()
    page_data = docs[0].page_content
 
    # Double braces are used to avoid prompt template variable interpretation
    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
 
        ### INSTRUCTION:
        Extract the job posting details. Provide the answer strictly as a JSON object with the following keys:
        `role`, `experience`, `skills`, `description`.
 
        No extra text outside of the JSON. No preamble, no trailing text.
 
        If you do not find all fields, use empty strings.
 
        ### OUTPUT FORMAT:
        {{
          "role": "...",
          "experience": "...",
          "skills": "...",
          "description": "..."
        }}
        """
    )
 
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_data})
 
    print("LLM Response:", res.content)
 
    try:
        job_info = json.loads(res.content)
        return job_info
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        print("LLM returned:", repr(res.content))
        return None
 
# Step 4: Read the reference resume PDF
def read_resume_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text
    return resume_text
 
# Step 5: Customize the resume to match the job requirements
def customize_resume(resume_text, job_requirements):
    prompt_modify = PromptTemplate.from_template(
        """
        ### ORIGINAL RESUME:
        {resume_text}
 
        ### JOB REQUIREMENTS:
        {job_requirements}
 
        ### INSTRUCTION:
        You are a professional resume writer. Modify the original resume to better match the job requirements while maintaining honesty and integrity.
        Highlight relevant skills and experience, rephrase where necessary, and ensure the resume is formatted professionally.
        Do not include any false information.
 
        ### UPDATED RESUME (NO PREAMBLE):
        """
    )
 
    chain_modify = prompt_modify | llm
    res = chain_modify.invoke({
        'resume_text': resume_text,
        'job_requirements': json.dumps(job_requirements)
    })
 
    return res.content
 
# Step 6: Save the updated resume as PDF using fpdf2 with Unicode Support
def save_resume_as_pdf(resume_text, output_pdf_path):
    # Using fpdf2 for better Unicode support
    pdf = FPDF()
    pdf.add_page()
 
    # Add a Unicode-supported TrueType font
    # Make sure DejaVuSans.ttf is available in the current directory
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
 
    pdf.set_auto_page_break(auto=True, margin=15)
 
    # fpdf2 supports Unicode if a TrueType font is used with uni=True
    # We can write the text directly
    pdf.multi_cell(0, 10, resume_text)
 
    pdf.output(output_pdf_path)
 
# Step 7: Main Execution
if __name__ == "__main__":
    # Replace with the actual job posting URL
    job_url = "https://jobs.nike.com/job/R-42315?from=job%20search%20funnel"
    job = extract_job_requirements(job_url)
 
    if job is None:
        print("Could not extract job requirements. Please check the URL and try again.")
    else:
        print("Extracted Job Requirements:", job)
 
        # Replace with your sample resume PDF path
        reference_resume_path = "Resume_John_Doe.pdf"
        original_resume = read_resume_pdf(reference_resume_path)
        print("Original Resume Extracted.")
 
        updated_resume = customize_resume(original_resume, job)
        print("Updated Resume Generated.")
 
        updated_resume_path = "updated_resume.pdf"
        save_resume_as_pdf(updated_resume, updated_resume_path)
        print("Updated resume saved at:", updated_resume_path)
