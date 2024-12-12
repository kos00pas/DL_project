import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import re
import fitz  # PyMuPDF for PDF handling
import nltk
from nltk.corpus import stopwords
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

# -----------------------------------------------------------------------------------

# NLTK and spaCy downloads
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)
nlp = spacy.load("en_core_web_sm")
print(nlp)
high_level_keywords = ["objective", "goal", "approach", "propose", "demonstrate", "result", "outcome",
                       "find", "discover", "introduce", "overview", "conclude", "describe", "present", "model"]

# -----------------------------------------------------------------------------------

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

def filter_summary(summary_text):
    # Filter out repetitive or incomplete sentences in summary
    sentences = summary_text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences)

# -----------------------------------------------------------------------------------


def generate_summary(text, model, tokenizer, min_words=200, max_words=300):
    # Generate summary using BART model, aiming for specified word length
    # Set min and max length in tokens, estimating 1.3 tokens per word on average
    min_length = int(min_words * 1.3)
    max_length = int(max_words * 1.3)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=5,
        length_penalty=1.5,  # Balanced to prevent overly terse or verbose summaries
        no_repeat_ngram_size=3,  # Avoids repetitive phrases for readability
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# -----------------------------------------------------------------------------------


def comparisons(abstract, summary):
    if not abstract or abstract == "Abstract not found.":
        return "Abstract not available for comparison."

    # Calculate basic similarity by overlapping words
    abstract_words = set(abstract.lower().split())
    summary_words = set(summary.lower().split())
    common_words = abstract_words.intersection(summary_words)
    relevance_score = len(common_words) / len(abstract_words) if abstract_words else 0
    relevance_percentage = relevance_score * 100

    return f"Relevance Score: {relevance_percentage:.2f}% - Common words: {len(common_words)} / {len(abstract_words)}"


# -----------------------------------------------------------------------------------

def extract_abstract(pdf_path):
    try:
        doc = fitz.open(pdf_path) # fitz library (PyMuPDF) to open the PDF file for processing
        text = "".join([doc.load_page(i).get_text("text") for i in range(min(2, doc.page_count))])  # Use only the first 2 pages for Abstract
        doc.close()

        # Regex pattern for Abstract section, allowing some variation
        stop_keywords = ["Introduction", "Background", "Conclusion", "References", "Acknowledgments", "Keywords","Related Work","$"]
        abstract_pattern = rf"(?:^|\n)\s*(Abstract|Summary|Overview|Synopsis)\s*[:.\-]?\s*(.*?)(?=\n\s*({'|'.join(stop_keywords)}))"

        # (?:^|\n) matches the start of the string or a newline, ensuring we find "Abstract" at the beginning of a line.
        # \s*(Abstract) matches the word "Abstract" with optional whitespace before and after.
        # [:.\-]? accounts for different formats after "Abstract" (e.g., "Abstract:", "Abstract.", or "Abstract-").
        # (.*?) captures the actual abstract text (non-greedy to stop at the next section).
        # (?=\n\s*(Introduction|Background|Conclusion|References|$)) specifies that the abstract ends where the next major section starts or at the end of the document.

        # Extract abstract based on pattern match
        abstract_match = re.search(abstract_pattern, text, re.S | re.I)
        # re.S allows the dot (.) to match newline characters.
        # re.I makes the search case-insensitive.

        abstract_text = clean_text(abstract_match.group(2)) if abstract_match else "Abstract not found."
        # abstract_match.group(2) retrieves the actual abstract text captured by the (.*?) group.
        # clean_text() is applied to clean up the extracted text (e.g., removing extra whitespace or unwanted characters).

        # Print debug information for extracted abstract
        print(f"\n[DEBUG] Abstract for {os.path.basename(pdf_path)}:\n{abstract_text}\n{'-'*80}")

        return abstract_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def extract_and_summarize(pdf_path, model, tokenizer):
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

        # Clean and filter based on matches or fallback to default
        introduction_text = clean_text(intro_match.group(2)) if intro_match else clean_text(text[:int(len(text) * 0.3)])
        conclusion_text = clean_text(concl_match.group(2)) if concl_match else clean_text(text[-int(len(text) * 0.3):])
        problem_and_solution_text = (clean_text(problem_and_solution_match.group(2))
                                      if problem_and_solution_match else "No problem or solution section found.")

        # Print debug information for extracted sections
        print(f"\n[DEBUG] Introduction for {os.path.basename(pdf_path)}:\n{introduction_text}\n{'-'*80}")
        print(f"\n[DEBUG] Conclusion for {os.path.basename(pdf_path)}:\n{conclusion_text}\n{'-'*80}")
        print(f"\n[DEBUG] Problem and Solution for {os.path.basename(pdf_path)}:\n{problem_and_solution_text}\n{'-' * 80}")

        # Combine extracted sections and generate summary
        if introduction_text and conclusion_text:
            combined_text = f"Introduction: {introduction_text}\nProblem and Solution: {problem_and_solution_text}\nConclusion: {conclusion_text}"
            summary = generate_summary(combined_text, model, tokenizer)
            return filter_summary(summary)

        return None

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# -----------------------------------------------------------------------------------

def summarize_and_compare_pdfs(pdf_directory):
    print("summarize_and_compare_pdfs")
    model_name = "facebook/bart-large-cnn"
    #  pre-trained model provided by Facebook's AI team,
    #  specifically a variant of the BART (Bidirectional and Auto-Regressive Transformers) model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # BartForConditionalGeneration is a class provided by the Hugging Face library
    # that is tailored for tasks like text generation and summarization.
    tokenizer = BartTokenizer.from_pretrained(model_name)
    # A tokenizer converts raw text into token IDs, which are numeric representations that the model can understand.
    print("start for")
    for filename in os.listdir(pdf_directory):
        print(filename)
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)

            # Extract abstract
            abstract = extract_abstract(pdf_path)
            print(f"\n[DEBUG] Abstract for {filename}:\n{abstract}\n{'-'*80}")

            # Generate enhanced summary
            enhanced_summary = extract_and_summarize(pdf_path, model, tokenizer)
            print(f"\n[DEBUG] Enhanced Summary for {filename}:\n{enhanced_summary}\n{'-'*80}")

            # Compare abstract and enhanced summary
            if enhanced_summary:
                comparison_result = comparisons(abstract, enhanced_summary)
                print(f"[DEBUG] Comparison for {filename}:\n{comparison_result}\n{'-'*80}")

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# Run the summarization and comparison block
pdf_directory = "./10"
summarize_and_compare_pdfs(pdf_directory)