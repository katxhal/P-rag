from ollama import Client

# Initialize the Ollama client to communicate with the local model
client = Client(host='http://localhost:11434')

def get_response_from_ollama(query):
    # Send the query to the Ollama model and get the response
    response = client.chat(model='dolphin-mistral:latest', messages=[{'role': 'user', 'content': query}])
    return response['message']['content']

def main():
    while True:
        # Take the query from the user
        user_query = input("Enter your query (or type 'exit' to stop): ")
        if user_query.lower() == 'exit':
            break
        
        # Get the response from Ollama and print it to the console
        answer = get_response_from_ollama(user_query)
        print("Ollama's response:", answer)

if __name__ == "__main__":
    main()
