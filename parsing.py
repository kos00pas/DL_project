import os
import re
import fitz  # PyMuPDF for PDF handling
import spacy
import shutil

from win32print import PRINTER_ENUM_NAME


pdf_directory = "./10"
output_directory = "./10"
nlp = spacy.load("en_core_web_sm")
high_level_keywords = ["objective", "goal", "approach", "propose", "demonstrate", "result", "outcome",
                       "find", "discover", "introduce", "overview", "conclude", "describe", "present", "model"]
folder_counter = 0

def clean_text(text):
    # Text cleaning and sentence filtering function
    # Basic text cleaning
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\([A-Za-z,0-9\s&;]*\)', '', text)
    text = re.sub(r'\[[0-9,; ]+\]', '', text)
    text = re.sub(r'(Figure|Table|Equation|Eq|Fig|Exhibit)\s*\d+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Segment into sentences and filter by length and keywords
    doc = nlp(text)
    sentences = [
        sent.text.strip() for sent in doc.sents
        if len(sent.text.split()) > 5 and any(keyword in sent.text.lower() for keyword in high_level_keywords)
    ]

    # Limit to a concise number of sentences
    return ' '.join(sentences[:5])
def extract_abstract(pdf_path):
    try:
        doc = fitz.open(pdf_path)  # Open the PDF file
        text = "".join([doc.load_page(i).get_text("text") for i in range(min(2, doc.page_count))])  # Use only the first 2 pages
        doc.close()

        # Regex pattern for Abstract section
        stop_keywords = ["Introduction", "Background", "Conclusion", "References", "Acknowledgments", "Keywords", "Related Work", "$"]
        abstract_pattern = rf"(?:^|\n)\s*(Abstract|Summary|Overview|Synopsis)\s*[:.\-]?\s*(.*?)(?=\n\s*({'|'.join(stop_keywords)}))"

        # Match abstract
        abstract_match = re.search(abstract_pattern, text, re.S | re.I)

        if abstract_match:
            abstract_text = clean_text(abstract_match.group(2))  # Extract and clean the abstract
            return abstract_text
        else:
            return None  # Return None if no abstract is found

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def extracted_infos(pdf_path):
    try:
        # Load PDF and extract text
        doc = fitz.open(pdf_path)
        text = "".join([doc.load_page(i).get_text("text") for i in range(doc.page_count)])
        doc.close()

        # Regex patterns for relevant sections
        intro_pattern = (r"(?:^|\n)"
                         r"\s*(Introduction|Objective|Background)"
                         r"\s*[:.\-]?"
                         r"\s*(.*?)"
                         r"(?=\n\s*(Conclusion|Summary|Results|References|Acknowledgments|Appendix|$))")

        concl_pattern = (r"(?:^|\n)"
                         r"\s*(Conclusion|Summary|Closing Remarks|DISCUSSION & CONCLUSIONS|Final Thoughts)"
                         r"\s*[:.\-]?"
                         r"\s*(.*?)"
                         r"(?=\n\s*(References|Acknowledgments|Appendix|$))")

        problem_and_solution_pattern = (
            r"(?:^|\n)\s*(Problem Statement|Challenges|Motivation|Existing Gaps|Contributions|Our Approach|Proposed Solution|Novelty|Innovative Approach)"
            r"\s*[:.\-]?\s*(.*?)"
            r"(?=\n\s*(Method|Results|Discussion|Conclusion|References|Acknowledgments|Appendix|$))"
        )

        # Extract matches within relevant parts of text
        intro_match = re.search(intro_pattern, text[:int(len(text) * 0.3)], re.S | re.I)
        concl_match = re.search(concl_pattern, text[-int(len(text) * 0.3):], re.S | re.I)
        problem_and_solution_match = re.search(problem_and_solution_pattern, text, re.S | re.I)

        # Extract section text or fallback to defaults
        introduction_text = clean_text(intro_match.group(2)) if intro_match else clean_text(text[:int(len(text) * 0.3)])
        conclusion_text = clean_text(concl_match.group(2)) if concl_match else clean_text(text[-int(len(text) * 0.3):])
        problem_and_solution_text = (clean_text(problem_and_solution_match.group(2))
                                      if problem_and_solution_match else "No problem or solution section found.")

        # Improved Binary Indicators
        intro_found = 1 if len(introduction_text.strip()) > 50 else 0  # Introduction should have enough text
        concl_found = 1 if len(conclusion_text.strip()) > 50 else 0  # Conclusion should have enough text
        problem_and_solution_found = 1 if len(problem_and_solution_text.strip()) > 50 else 0  # Problem & solution check

        # Debugging outputs
        print(f"Introduction Found: {intro_found}, Conclusion Found: {concl_found}, Problem/Solution Found: {problem_and_solution_found}")

        # Combine extracted sections and generate summary
        combined_text = f"Introduction: {introduction_text}\nProblem and Solution: {problem_and_solution_text}\nConclusion: {conclusion_text}"

        # Return binary array and combined text
        return [intro_found, problem_and_solution_found, concl_found], combined_text

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return [0, 0, 0], None  # Return [0, 0, 0] and None in case of an error




success_list = []
skipping_list = []

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        # print(filename)

        # Ensure folder does not exist from a previous run
        folder_name = os.path.join(output_directory, str(folder_counter))
        while os.path.exists(folder_name):
            folder_counter += 1  # Increment counter until a non-existing folder is found
            folder_name = os.path.join(output_directory, str(folder_counter))

        os.makedirs(folder_name)

        pdf_path = os.path.join(pdf_directory, filename)

        # Copy the PDF file to the new folder
        destination_pdf_path = os.path.join(folder_name, filename)
        shutil.copy(pdf_path, destination_pdf_path)
        # print(f"Created folder: {folder_name} : Copied {filename} to {destination_pdf_path}")

        # ------------------
        abstract = extract_abstract(pdf_path)
        if abstract is None:
            print(f"Abstract not found in {filename}. Skipping...")
            skipping_list.append(folder_counter)
            continue

        abstract_file_path = os.path.join(folder_name, "abstract.txt")
        with open(abstract_file_path, "w", encoding="utf-8") as abstract_file:
            abstract_file.write(abstract)
        # ------------------


        # ------------------
        binary_array, infos = extracted_infos(pdf_path)
        if binary_array[0] == 0 and binary_array[2] == 0:
            print(f"Introduction and Conclusion missing in {filename}. Skipping...")
            print(binary_array)
            print(infos)
            skipping_list.append(folder_counter)
            continue


        info_file_path = os.path.join(folder_name, "infos.txt")
        with open(info_file_path, "w", encoding="utf-8") as info_file:
            info_file.write(infos)
        # ------------------
        success_list.append(folder_counter)
        folder_counter += 1


print(skipping_list)