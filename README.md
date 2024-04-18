# Katxhal RAG App

Katxhal RAG App is a Streamlit application that allows you to query PDF files or URLs using a Retrieval-Augmented Generation (RAG) approach. It combines the power of large language models (LLMs) with external data sources to provide accurate and context-aware responses.

## Features

- Query PDF files or URLs for information
- Retrieval-Augmented Generation (RAG) for context-aware responses
- Option to use Perplexity API for generating responses
- Streamlit-based user interface for easy interaction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/katxhal-rag-app.git
cd katxhal-rag-app
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:

```bash
streamlit run app/main.py
```

2. The app will open in your default web browser. You can upload a PDF file or enter URLs (one per line) in the provided input fields.

3. Enter your question in the chat input box.

4. If you want to use the Perplexity API, toggle the "Use Perplexity API" option in the sidebar and enter your Perplexity API key.

5. The app will process your input and display the generated response in the chat window.

## Configuration

You can configure the app by modifying the following settings in the `app/main.py` file:

- `models`: List of available models for the Ollama LLM.
- `selected_model`: Default model to use for generating responses.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the [Apache License](LICENSE).