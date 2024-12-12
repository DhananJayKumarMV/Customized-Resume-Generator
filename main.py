# Step 1: Import Necessary Libraries
from langchain.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import Dict
import json
import PyPDF2
from fpdf import FPDF

# Define the categories to extract from the resume
CATEGORIES = ["Name", "Education", "Experience", "Skills", "Certifications", "Contact Information"]

# Step 2: Initialize ChromaDB
def initialize_chromadb():
    client = chromadb.PersistentClient(path="vector_db")  # Persistent storage for the database
    collection = client.get_or_create_collection(name="portfolio")  # "portfolio" is the collection name
    return collection


def read_resume_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text
    return resume_text

# Step 3: Load CSV Data into ChromaDB
def load_csv_to_chromadb(collection, csv_path):
    """
    Load data into ChromaDB from a CSV file for role matching.
    The CSV file should have columns: Role, Skills, Experience, Projects.
    """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        collection.add(
            documents=[json.dumps(row.to_dict())],
            metadatas={"Role": row["Role"]},
            ids=[str(uuid.uuid4())]  # Generate a unique ID for each row
        )


# Step 4: Query ChromaDB Based on Exact Role Match
def query_chromadb(collection, job_role):
    """
    Query the ChromaDB collection for the most relevant data based on an exact role match.
    """
    results = collection.get(
        where={"Role": job_role}  # Filter by role using the 'where' clause
    )

    if results and "documents" in results and results["documents"]:
        first_result = results["documents"][0]
        if isinstance(first_result, str):  # If the result is a JSON string, parse it
            return json.loads(first_result)
    return None  # Return None if no valid result is found



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
 
# Step 1: Extract category-specific information
def extract_category_info(resume_text: str, category: str) -> str:
    """
    Extracts information for a specific category from the resume using a language model.
    """
    prompt_category = PromptTemplate.from_template(
        f"""
        ### ORIGINAL RESUME TEXT:
        {{resume_text}}

        ### INSTRUCTION:
        Extract the {category.lower()} of the individual from the resume. Provide only the {category.lower()},
        strictly as a string, without any preamble or additional text.
        """
    )
    
    chain_category = prompt_category | llm
    res_category = chain_category.invoke(input={"resume_text": resume_text})

    print(f"LLM Response for {category}:", res_category.content)

    return res_category.content.strip()

# Step 2: Create a function to identify and extract all categories
def extract_all_categories(resume_text: str) -> Dict[str, str]:
    """
    Extracts all predefined categories from the resume.
    """
    extracted_data = {}
    for category in CATEGORIES:
        extracted_data[category] = extract_category_info(resume_text, category)
    return extracted_data

# Step 3: Save extracted information to JSON
def save_to_json(data: Dict, output_path: str):
    """
    Saves extracted data to a JSON file.
    """
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data saved to {output_path}")

# Step 4: Customize the resume using extracted data and job requirements
def customize_resume_with_json(resume_text: str, job_requirements: Dict, extracted_data_path: str):
    """
    Customize the resume based on extracted category information and job requirements.
    """
    with open(extracted_data_path, "r") as json_file:
        extracted_data = json.load(json_file)

    prompt_customize = PromptTemplate.from_template(
        """
        ### ORIGINAL RESUME:
        {resume_text}

        ### JOB REQUIREMENTS:
        {job_requirements}

        ### EXTRACTED DATA:
        {extracted_data}

        ### INSTRUCTION:
        You are a professional resume writer. Modify the original resume to better match
        the job requirements while maintaining honesty and integrity. Use the extracted data
        to ensure accuracy and enhance the relevancy of the resume.
        Highlight relevant skills and experience, rephrase where necessary, and ensure
        the resume is formatted professionally.

        ### UPDATED RESUME (NO PREAMBLE):
        """
    )

    chain_customize = prompt_customize | llm
    res_customize = chain_customize.invoke({
        "resume_text": resume_text,
        "job_requirements": json.dumps(job_requirements),
        "extracted_data": json.dumps(extracted_data)
    })

    return res_customize.content
 
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
    # Initialize ChromaDB
    collection = initialize_chromadb()

    # Path to the CSV file
    csv_path = "/content/drive/MyDrive/LLM project/Roles.csv"
    load_csv_to_chromadb(collection, csv_path)


    # Extract job requirements
    job_url = "https://jobs.nike.com/job/R-42315?from=job%20search%20funnel"
    job = extract_job_requirements(job_url)
 
    if job:
        print("Extracted Job Requirements:", job)

        # Query the vector database based on an exact role match
        db_record = query_chromadb(collection, job["role"])

        if db_record:
            print("Matching Record from DB:", db_record)

    print(job)
 
    reference_resume_path = "/content/drive/MyDrive/LLM project/Resume_John_Doe.pdf"
    original_resume = read_resume_pdf(reference_resume_path)
    print("Original Resume Extracted.")

    # Extract all categories
    # extracted_data = extract_all_categories(example_resume_text)
    extracted_data = extract_all_categories(original_resume)
    

    # Save extracted data to JSON
    extracted_data_path = "/content/drive/MyDrive/LLM project/extracted_resume_data.json"
    save_to_json(extracted_data, extracted_data_path)

    # Customize the resume
    # updated_resume = customize_resume_with_json(original_resume, example_job_requirements, extracted_data_path)
    updated_resume = customize_resume_with_json(original_resume, db_record, extracted_data_path)
        # print("Matching Record from DB:", db_record)

    print("\nUpdated Resume:\n", updated_resume)

    updated_resume_path = "/content/drive/MyDrive/LLM project/updated_resume.pdf"
    save_resume_as_pdf(updated_resume, updated_resume_path)
    print("Updated resume saved at:", updated_resume_path)