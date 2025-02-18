from langchain_ollama import ChatOllama
import PyPDF2
import docx
import os
import json
import logging
import random
from io import BytesIO
import pandas as pd
from flask import Flask, request, send_file, jsonify
from flask_restx import Api, Resource, fields
app = Flask(__name__)
api = Api(app, version='1.0', title='Resume Ranking API',
          description='API to extract ranking criteria from a job description and score resumes against those criteria.')

ns = api.namespace('api', description='Resume Ranking operations')

# Configure logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the expected model for the /extract-criteria endpoint using Swagger
extract_model = api.parser()
extract_model.add_argument('file', location='files', type='file', required=True, help='Job description file (PDF or DOCX)')

# Define the expected model for the /score-resumes endpoint using Swagger
score_model = api.parser()
score_model.add_argument('criteria', location='form', required=True, help='JSON string representing a list of criteria')
#score_model.add_argument('files', location='files', type='file', action='append', required=True, help='One or more resume files (PDF or DOCX)')
score_model.add_argument('file', location='files', type='file', required=True, help='One or more resume files (PDF or DOCX)')


def extract_text(file_storage):
    """
    Determines the file type (PDF or DOCX) and extracts text accordingly.
    :param file_storage: A werkzeug FileStorage object.
    :return: Extracted text.
    """
    filename = file_storage.filename.lower()
    logger.debug(f"Processing file: {filename}")
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_storage)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file_storage)
    else:
        logger.error("Unsupported file format.")
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

def extract_text_from_pdf(file_stream):
    """
    Extracts text from a PDF file.
    :param file_stream: A file-like object containing PDF data.
    :return: Extracted text as a string.
    """
    logger.debug("Extracting text from PDF file.")
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        logger.debug(f"Extracted text from page {page_num}: {page_text[:50]}...")
        text += page_text + "\n"
    return text

def extract_text_from_docx(file_stream):
    """
    Extracts text from a DOCX file.
    :param file_stream: A file-like object containing DOCX data.
    :return: Extracted text as a string.
    """
    logger.debug("Extracting text from DOCX file.")
    doc = docx.Document(file_stream)
    text = "\n".join([para.text for para in doc.paragraphs])
    logger.debug(f"Extracted DOCX text: {text[:50]}...")
    return text
    print(text)

llm = ChatOllama(
    model="llama",
    temperature=0,
    base_url="http://0.0.0.0:11434",
    format="json",
    # other params...
)

def get_job_criteria(text):
    messages = [
        (
        "You are an expert in extracting job requirements from job descriptions. Your task is to identify and extract the key criteria from the given job description and format them as a JSON object. Please ensure that each criterion is clear, specific, and directly mentioned in the job description.Return the extracted criteria in the following JSON format:"
        "{\n"
        '  "criteria": [\n'
        '    "Criteria 1",\n'
        '    "Criteria 2",\n'
        '    "Criteria 3"\n'
        "  ]\n"
        "}\n\n"
        ),
    ("human", "Job Description:\n" + text),
    ]
    print(messages)
    resp = llm.invoke(messages)
    return resp.content
    print(resp.content)

@ns.route('/extract-criteria')
class ExtractCriteria(Resource):
    @ns.expect(extract_model)
    @ns.response(200, 'Success')
    def post(self):
        """
        Extract ranking criteria from a job description file.
        ---
        Consumes:
            - multipart/form-data
        Parameters:
            - in: formData
              name: file
              type: file
              required: true
              description: The job description file (PDF or DOCX).
        Produces:
            - application/json
        Responses:
            200: A JSON object containing a list of extracted criteria.
        """
        try:
            # Debug: Log the incoming request
            logger.debug("Received request for /extract-criteria endpoint.")
            uploaded_file = request.files.get('file')
            if not uploaded_file:
                logger.error("No file part in the request.")
                return {"message": "No file provided."}, 400

            # Extract text from the provided file
            text = extract_text(uploaded_file)
            logger.debug(f"Extracted text: {text[:100]}...")

            # Simulate LLM extraction of ranking criteria
            criteria = get_job_criteria(text)
            # Debug: Log the output criteria
            logger.debug(f"Returning extracted criteria: {criteria}")

            #return {"criteria": criteria}, 200
            return criteria
        except Exception as e:
            logger.exception("Error processing /extract-criteria request.")
            return {"message": str(e)}, 500

def get_resume_score(criteria_json_string, text):
    messages = [
        (
        "You are an expert in evaluating resumes. First extract the name of the candidate from the resume. Then extract each of the criterias. Using the given criterias and the resume, assess the candidate's qualifications on each criterion, scoring between 0 and 10. A score of 0 indicates a poor match, and a score of 10 indicates an excellent match. Do not copy the example given.Provide the assessment as a JSON object in the following format:"
        "{\n"
        '  "Candidate Name": "Name of the candidate" [\n'
        '    "put name of the first criteria extracted here": put score here,\n'
        '    "put name of the second criteria extracted here": put score here,\n'
        '    "put name of the third criteria extracted here": put score here,\n'
        '     .....    \n'
        "  ]\n"
        "}\n\n"
        ),
    ("human", criteria_json_string + "\nResume:\n" + text),
    ]
    print(messages)
    resp = llm.invoke(messages)
    return resp.content
    print(resp.content)
@ns.route('/score-resumes')
class ScoreResumes(Resource):
    @ns.expect(score_model)
    @ns.response(200, 'Success')
    def post(self):
        """
        Score resumes against provided ranking criteria.
        ---
        Consumes:
            - multipart/form-data
        Parameters:
            - in: formData
              name: criteria
              type: string
              required: true
              description: A JSON string representing the list of ranking criteria.
            - in: formData
              name: files
              type: file
              required: true
              description: One or more resume files (PDF or DOCX).
        Produces:
            - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        Responses:
            200: An Excel sheet with candidate names, individual scores, and total scores.
        """

        try:
            logger.debug("Received request for /extract-criteria endpoint.")
            uploaded_file = request.files.get('file')
            if not uploaded_file:
                logger.error("No file part in the request.")
                return {"message": "No file provided."}, 400

            # Extract text from the provided file
            text = extract_text(uploaded_file)
            logger.debug(f"Extracted text: {text[:100]}...")

            criteria_str = request.form.get('criteria')

            #if not uploaded_files:
                #logger.error("No resume files provided.")
                #return {"message": "No resume files provided."}, 400

            result = get_resume_score(criteria_str, text)

            # Create a DataFrame and convert it to an Excel file in memory
            #return results
            #return result
            data = json.loads(result)
            total_score = sum(value for key, value in data.items() if isinstance(value, (int, float)))
            data["total score"] = total_score
            df = pd.DataFrame([data])
            logger.debug("Created DataFrame for results:")
            logger.debug(df.head())

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Scores')
            output.seek(0)

            logger.debug("Returning Excel file as response.")
            return send_file(output, download_name="resume_scores.xlsx", as_attachment=True,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except Exception as e:
            logger.exception("Error processing /score-resumes request.")
            return {"message": str(e)}, 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
