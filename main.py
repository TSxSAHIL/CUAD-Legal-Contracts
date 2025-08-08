import os
import json
import fitz  
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from tqdm import tqdm
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please create a .env file and add your key.")

genai.configure(api_key=GOOGLE_API_KEY)

CONTRACTS_DIR = "sample"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "contract_analysis.json")
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(OUTPUT_DIR, "clause_metadata.json")
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "gemini-2.5-pro"


class ContractAnalysis(BaseModel):
    """
    Pydantic model for structuring the data extracted from a contract.
    This ensures the LLM output is validated and fits a predictable structure.
    """
    summary: str = Field(
        description="A 100-150 word summary covering the agreement's purpose, key obligations of each party, and notable risks or penalties."
    )
    termination_clause: Optional[str] = Field(
        description="The full text of the clause(s) detailing conditions for contract termination. Should be null if not found."
    )
    confidentiality_clause: Optional[str] = Field(
        description="The full text of the confidentiality clause. Should be null if not found."
    )
    liability_clause: Optional[str] = Field(
        description="The full text of the clause limiting liability or the indemnity clause. Should be null if not found."
    )

FEW_SHOT_PROMPT_TEMPLATE = """
**INSTRUCTION:**
You are an expert legal assistant. Analyze the legal contract provided below and perform two tasks:
1.  **Summarize:** Write a concise 100-150 word summary covering the agreement's main purpose, each party's key obligations, and any significant risks or penalties.
2.  **Extract Clauses:** Extract the full text for the 'Termination', 'Confidentiality', and 'Liability/Indemnity' clauses. If a specific clause is not present, the value for that key must be null.

Your final output MUST be a single, valid JSON object, without any markdown formatting (`json ...`), comments, or other text outside the JSON structure.

--- START OF EXAMPLE ---

**Example Contract Text (from BerkshireHillsBancorpInc_20120809_10-Q_EX-10.16):**
"...9. CONFIDENTIALITY. Except as required by federal securities laws, or federal or state banking laws, each party agrees: (i) that it will not disclose to any third party or use any Confidential Information... 10. TERMINATION AND DEFAULT. a) TERMINATION FOR BREACH. Either Party shall have the right, without prejudice to any other rights it may have, to terminate this Agreement if the other Party materially breaches its obligation hereunder and such breach remains uncured... 11. TERMINATION BY BERKSHIRE. Berkshire may terminate this Agreement immediately by giving Auriemma notice if (i) Auriemma dies or is prevented by injury or illness from satisfactorily performing the obligations required by this Agreement; (ii) Auriemma is convicted of a felony or criminal offense involving dishonesty or fraud; or (iii) Auriemma publicly disparages Berkshire and/ or its products... 20. INDEMNITY AND INSURANCE. Berkshire shall indemnify and hold Auriemma harmless from and against any and all claims, actions, suits, proceedings, losses, damages and expenses (including, without limitation, reasonable attorneys , consultants' and experts' fees) (collectively, \"Claims\") arising out of or relating to any inaccuracy or breach of Berkshire's representations, warranties, covenants or any claim or other cause of action arising out of or in connection with this Agreement, including actions based upon gross negligence of Berkshire under this Agreement."

**Example JSON Output:**
{{
  "summary": "This Endorsement Agreement is between Geno Auriemma and Berkshire Bank for Auriemma's endorsement of Berkshire's financial services. Auriemma grants Berkshire the exclusive right to use his name and likeness for promotion within a defined territory and period. In return, Berkshire will pay Auriemma a total of $480,000 in cash and stock installments. Auriemma's primary obligations are to provide endorsement services and make specified appearances, while Berkshire's is to provide compensation. Risks include termination for breach, insolvency, or specific actions by Auriemma such as criminal conviction or disparagement. Berkshire indemnifies Auriemma against claims arising from the agreement.",
  "termination_clause": "a) TERMINATION FOR BREACH. Either Party shall have the right, without prejudice to any other rights it may have, to terminate this Agreement if the other Party materially breaches its obligation hereunder and such breach remains uncured. A material breach occurs if either Party (i) fails to make any payment, or (ii) fails to observe or perform any of the covenants, agreements, or obligations (other than payments of money). Upon the breach of either of the above conditions, the non-defaulting party may terminate this Agreement as follows: (A) as to a default under clause (i) above, if payment is not made within ten (10) days after the defaulting party shall have received written notice of such failure to make payment; or (B) as to a default under clause (ii) above, if such default is not cured within thirty (30) days after the defaulting party shall have received written notice specifying in reasonable detail the nature of such default and such action the defaulting party must take in order to cure each such item of default. b) TERMINATION DUE TO INSOLVENCY. If either Party (the \\\"Bankrupt Party\\\"), (i) commences or becomes the subject of any case or proceeding under the bankruptcy or insolvency laws; (ii) has appointed for it or for any substantial part of its property a court-appointed receiver, liquidator, assignee, trustee, custodian, sequestrator or other similar official; (iii) makes an assignment or the benefit of its credits; (iv) fails generally to pay its debts as they become due; or (v) takes corporate action in furtherance of any of the foregoing (collectively, herein referred to as \\\"Events of Insolvency\\\"), then, in each case, the Bankrupt Party shall immediately give notice of such event to the other Party. Whether or not such notice is given, the other Party shall have the right, to the fullest extent permitted under applicable law, following the occurrence of any Event of Insolvency and without prejudice to any other rights it may have, at any time thereafter to terminate this Agreement, effective immediately upon giving notice to the Bankrupt Party. Berkshire may terminate this Agreement immediately by giving Auriemma notice if (i) Auriemma dies or is prevented by injury or illness from satisfactorily performing the obligations required by this Agreement; (ii) Auriemma is convicted of a felony or criminal offense involving dishonesty or fraud; or (iii) Auriemma publicly disparages Berkshire and/ or its products.",
  "confidentiality_clause": "Except as required by federal securities laws, or federal or state banking laws, each party agrees: (i) that it will not disclose to any third party or use any Confidential Information, as defined herein, disclosed to it by the other party except as expressly permitted in this Agreement and (ii) that it will take all reasonable measures to maintain the confidentiality of all Confidential Information of the other party in its possession or control, which will in no event be less than the measures it uses to maintain the confidentiality of its own information of similar importance. For the purpose of this Agreement, Confidential Information shall mean all information, materials and data, in any form, format or medium, disclosed, or revealed to either party in any way relating to the other party's business including but not limited to its finances, customers, operations, products, services, plans, pricing, suppliers, business strategies or any other similar information. Confidential Information may be contained in written material, verbal or electronic communications.",
  "liability_clause": "Berkshire shall indemnify and hold Auriemma harmless from and against any and all claims, actions, suits, proceedings, losses, damages and expenses (including, without limitation, reasonable attorneys , consultants' and experts' fees) (collectively, \\\"Claims\\\") arising out of or relating to any inaccuracy or breach of Berkshire's representations, warranties, covenants or any claim or other cause of action arising out of or in connection with this Agreement, including actions based upon gross negligence of Berkshire under this Agreement. provided that Berkshire shall be given prompt notice of any such action or claim."
}}
--- END OF EXAMPLE ---

**Contract to Analyze:**
{contract_text}

**JSON Output:**
"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """This function extracts and normalizes text from a PDF file using PyMuPDF/fitz."""
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return " ".join(text.replace("\n", " ").split())
    except Exception as e:
        logging.error(f"Error reading or processing {pdf_path}: {e}")
        return ""

def analyze_contract_with_gemini(text: str) -> Optional[ContractAnalysis]:
    """This function sends contract text to the Gemini API and validates the JSON response."""
    if not text:
        return None

    model = genai.GenerativeModel(GENERATION_MODEL)
    prompt = FEW_SHOT_PROMPT_TEMPLATE.format(contract_text=text)

    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        if json_text.startswith("```json"):
            json_text = json_text[7:-3].strip()
        
        data = json.loads(json_text)
        return ContractAnalysis(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Failed to parse or validate LLM response. Error: {e}")
        logging.debug(f"LLM Response Text:\n---\n{response.text[:1000]}...\n---")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini API call: {e}")
        return None


def create_and_save_semantic_index(clauses_metadata: List[dict]):
    """Generates embeddings for clauses and builds/saves a FAISS index."""
    if not clauses_metadata:
        logging.warning("No clauses were extracted to create a semantic index.")
        return

    logging.info(f"Generating embeddings for {len(clauses_metadata)} clauses...")
    
    clause_texts = [item['text'] for item in clauses_metadata]
    
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=clause_texts,
            task_type="RETRIEVAL_DOCUMENT",
            title="Contract Clauses"
        )
        embeddings = result['embedding']
        
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))

        logging.info(f"FAISS index created successfully with {index.ntotal} vectors.")

        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(clauses_metadata, f, indent=4)
        
        logging.info(f"FAISS index saved to {FAISS_INDEX_FILE}")
        logging.info(f"Clause metadata saved to {METADATA_FILE}")

    except Exception as e:
        logging.error(f"Failed to create or save semantic index: {e}")

def perform_semantic_search(query: str, k: int = 3):
    """Performs a natural language search against the saved FAISS index."""
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        logging.error("Index not found. Please run the main processing pipeline first.")
        return

    logging.info(f"\nPerforming semantic search for query: '{query}'")

    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        query_embedding_result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array([query_embedding_result['embedding']], dtype=np.float32)

        distances, indices = index.search(query_embedding, k)

        print("\n--- Top Search Results ---")
        if indices.size == 0:
            print("No results found.")
            return
            
        for i in range(min(k, len(indices[0]))):
            result_index = indices[0][i]
            matched_item = metadata[result_index]
            print(f"#{i+1}: Closest Match (Distance: {distances[0][i]:.4f})")
            print(f"  - Contract ID: {matched_item['contract_id']}")
            print(f"  - Clause Type: {matched_item['clause_type']}")
            print(f"  - Text: \"{matched_item['text'][:250]}...\"")
            print("-" * 25)

    except Exception as e:
        logging.error(f"An error occurred during semantic search: {e}")


def main():
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)

    if not os.path.isdir(CONTRACTS_DIR):
        logging.error(f"Contracts directory '{CONTRACTS_DIR}' not found. Please create it and add your PDFs.")
        return
    else:
        contract_files = [f for f in os.listdir(CONTRACTS_DIR) if f.lower().endswith('.pdf')]

    if not contract_files:
        logging.warning(f"No PDF files found to process in '{CONTRACTS_DIR}'.")
        return

    all_results = []
    all_clauses_for_embedding = []

    logging.info(f"Starting contract analysis for {len(contract_files)} PDF(s)...")

    for filename in tqdm(contract_files, desc="Processing Contracts"):
        contract_id = os.path.splitext(filename)[0]
        pdf_path = os.path.join(CONTRACTS_DIR, filename)

        logging.info(f"Extracting text from {filename}...")
        full_text = extract_text_from_pdf(pdf_path)

        if not full_text:
            continue

        logging.info(f"Analyzing contract {contract_id} with Gemini...")
        analysis = analyze_contract_with_gemini(full_text)

        if analysis:
            result_data = {"contract_id": contract_id, **analysis.model_dump()}
            all_results.append(result_data)

            if analysis.termination_clause:
                all_clauses_for_embedding.append({"contract_id": contract_id, "clause_type": "termination", "text": analysis.termination_clause})
            if analysis.confidentiality_clause:
                all_clauses_for_embedding.append({"contract_id": contract_id, "clause_type": "confidentiality", "text": analysis.confidentiality_clause})
            if analysis.liability_clause:
                all_clauses_for_embedding.append({"contract_id": contract_id, "clause_type": "liability", "text": analysis.liability_clause})
        else:
            logging.warning(f"Could not analyze {contract_id}.")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"\nAnalysis complete. Main results saved to {OUTPUT_FILE}")

    with open(METADATA_FILE, 'w') as f:
        json.dump(all_clauses_for_embedding, f, indent=4)
    logging.info(f"Clause metadata saved to {METADATA_FILE}")
    create_and_save_semantic_index(all_clauses_for_embedding)

    if os.path.exists(FAISS_INDEX_FILE):
        perform_semantic_search(query="What happens if confidential information is disclosed?")
        perform_semantic_search(query="How can the agreement be ended due to non-payment?")
        perform_semantic_search(query="Who is responsible for claims and damages?")

if __name__ == "__main__":
    main()