\# TruthLens — AI-Powered Fact-Checking System



TruthLens is an AI-powered fact-checking project designed to evaluate textual claims using evidence retrieval, Natural Language Inference (NLI), real-time web search, and AI-generated explanations.



The project combines historical evidence with online information to help users investigate claims and understand the evidence behind a verdict.



\## Features



\- \*\*Evidence Retrieval:\*\* Retrieves relevant evidence from the FactDrill dataset using semantic and keyword-based retrieval.

\- \*\*Semantic Search:\*\* Uses FAISS for vector-based evidence retrieval.

\- \*\*Hybrid Retrieval:\*\* Combines keyword-based and semantic retrieval approaches.

\- \*\*NLI Verification:\*\* Analyzes relationships between claims and evidence, including entailment, contradiction, and neutral relationships.

\- \*\*Real-Time Evidence Retrieval:\*\* Searches the web for potentially relevant and recent information.

\- \*\*Conflict Resolution:\*\* Combines evidence and verification results to produce a final verdict.

\- \*\*AI-Generated Explanations:\*\* Uses Ollama to generate explanations for fact-checking results.

\- \*\*REST API:\*\* Provides a FastAPI backend.

\- \*\*Frontend:\*\* A React application built with Vite.



\## Tech Stack



\### Backend and AI

\- Python

\- FastAPI

\- PyTorch

\- Hugging Face Transformers

\- Sentence Transformers

\- FAISS

\- BM25

\- Pandas and NumPy

\- Beautiful Soup

\- DDGS

\- Ollama



\### Frontend

\- React

\- Vite

\- Tailwind CSS

\- Lucide React



\## Project Structure



```text

TruthLens/

├── FactDrill/                 # Original dataset files

├── semantic\_index/             # FAISS index and evidence mapping

├── semantic\_index\_test/        # Additional test index

├── splits/                     # Dataset splits

├── scripts/

│   ├── data\_preparation/

│   ├── evaluation/

│   ├── retrieval/

│   └── verification/

├── outputs/                    # Analysis and pipeline outputs

├── truthlens.py                # Main fact-checking pipeline

├── truthlens\_api.py            # FastAPI application

├── requirements.txt            # Python dependencies

├── requirements-api.txt        # API dependencies

├── .gitattributes              # Git LFS configuration

├── .gitignore

└── truthlens-frontend/         # React frontend

```



\## Prerequisites



Install the following before running the project:



\- Python and pip

\- Git

\- Git LFS

\- Node.js and npm

\- Ollama



The main application expects the Ollama model `llama3.2:3b`.



\## 1. Clone the Repository



Install Git LFS if you have not already installed it, then run:



```bash

git lfs install

git clone https://github.com/Vaishnavi2-dot/TruthLens1.git

cd TruthLens1

git lfs pull

```



Git LFS downloads the large files tracked through Git LFS, including the evidence corpus and semantic index files.



If a required file is missing after cloning, check the repository's LFS configuration and upload status.



\## 2. Set Up the Python Environment



Create and activate a virtual environment.



\### Windows PowerShell



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks activation, use Command Prompt:



```bat

.venv\\Scripts\\activate.bat

```



Install the dependencies:



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



The first run may download model files from Hugging Face. An internet connection may be required for initial model downloads and real-time web retrieval.



\## 3. Install and Run Ollama



Install Ollama from:



https://ollama.com/



Download the model used by the main application:



```bash

ollama pull llama3.2:3b

```



Verify that the model is available:



```bash

ollama list

```



Ensure Ollama is running before using features that generate AI explanations.



\## 4. Run the Backend API



From the project root, activate the Python virtual environment and run:



```bash

uvicorn truthlens\_api:app --reload

```



The API should be available at:



```text

http://127.0.0.1:8000

```



Open the interactive API documentation at:



http://127.0.0.1:8000/docs



Use the Swagger interface to inspect the available endpoints and test requests.



\## 5. Run the Frontend



Open a second terminal and navigate to the frontend directory:



```bash

cd truthlens-frontend

npm install

npm run dev

```



Open the local URL printed by Vite in the terminal.



Make sure the backend is running as well. The frontend and backend must use matching API URLs and compatible CORS settings.



\## 6. Main Runtime Files



The main application currently expects these files:



\- `factdrill\_evidence\_corpus.csv`

\- `semantic\_index/evidence.faiss`

\- `semantic\_index/evidence\_mapping.csv`



Keep these paths consistent with the configuration in `truthlens.py`.



The original FactDrill dataset, dataset splits, and evaluation outputs support research and experimentation. Some of these files can be regenerated using scripts in the `scripts/` directory, depending on the stage of the pipeline.



\## 7. Research and Development



The scripts are grouped by purpose:



\- `scripts/data\_preparation/` — dataset cleaning, unification, splitting, and index construction.

\- `scripts/retrieval/` — keyword retrieval, semantic retrieval, hybrid retrieval, and real-time evidence collection.

\- `scripts/verification/` — NLI, conflict resolution, temporal credibility, and explanation generation.

\- `scripts/evaluation/` — retrieval evaluation, dataset analysis, duplicate detection, and tuning.



Check each script's input paths and configuration before running it. Some scripts generate large CSV files.



\## Current Limitations



TruthLens is an ongoing research and development project. Retrieval relevance, verdict correctness, confidence calibration, source reliability, and explanation quality still require systematic evaluation.



A generated verdict should not be treated as definitive proof that a claim is true or false. Real-time sources may be incomplete, outdated, or unreliable.



\## Contributing



Contributions and improvements are welcome.



1\. Clone the repository.

2\. Create a separate branch for your changes.

3\. Install the dependencies.

4\. Test your changes locally.

5\. Submit a pull request or share your changes with the project maintainers.



Before adding large datasets or generated files, check `.gitignore`, `.gitattributes`, and Git LFS storage limits.



\## License



No license has been specified yet. Please obtain permission before redistributing the original dataset or other third-party materials.

