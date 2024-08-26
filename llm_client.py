import openai


class LLMClient:
    def __init__(self, client_url, api_key, model_name, temperature, history):
        self.client_url = client_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.history = history

        # Create client for llama.cpp server
        print("Creating llama.cpp LLM server client...")
        self.client = openai.OpenAI(
            base_url = self.client_url,
            api_key = self.api_key
        )


    # Create LLM chat completion text stream and yield chunks
    def prompt(self, prompt):
        self.history.append(prompt)

        completion = self.client.chat.completions.create(
        messages = self.history,
        model = self.model_name,
        temperature = self.temperature,
        stream = True,
        extra_body = {"cache_prompt": True}
        )

        for chunk in completion:
            chunk_message =  chunk.choices[0].delta.content
            if chunk_message is not None:
                yield chunk.choices[0].delta.content
