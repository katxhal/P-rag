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

Citations:
[1] https://pplx-res.cloudinary.com/image/upload/v1713467170/user_uploads/osyczwnwl/image.jpg
[2] https://www.pinecone.io/learn/retrieval-augmented-generation/
[3] https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/
[4] https://learn.microsoft.com/en-us/mem/intune/apps/apps-win32-add
[5] https://www.datacamp.com/blog/what-is-retrieval-augmented-generation-rag
[6] https://huggingface.co/datasets/RyokoAI/ShareGPT52K/blob/main/README.md
[7] https://hyperight.com/7-practical-applications-of-rag-models-and-their-impact-on-society/
[8] https://docs.divio.com/how-to/install-python-dependencies/
[9] https://www.databricks.com/glossary/retrieval-augmented-generation-rag
[10] https://aws.amazon.com/what-is/retrieval-augmented-generation/
[11] https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/php-configuration-composer.html
[12] https://docs.flutter.dev/packages-and-plugins/using-packages
[13] https://huggingface.co/TheBloke/wizard-vicuna-13B-GPTQ/blob/main/README.md
[14] https://stackoverflow.com/questions/6024027/how-do-i-package-and-run-a-simple-command-line-application-with-dependencies-usi
[15] https://stackoverflow.com/questions/25654845/how-can-i-create-a-text-box-for-a-note-in-markdown
[16] https://www.jetbrains.com/help/hub/markdown-syntax.html
[17] https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app
[18] https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc
[19] https://github.com/f/awesome-chatgpt-prompts/blob/main/README.md
[20] https://packaging.python.org/en/latest/tutorials/managing-dependencies/