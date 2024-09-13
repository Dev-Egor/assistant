import json
import sys
import re
import functions
from initializer import Initializer


# Main assistant logic
def main():
    llm_client, tts_client, stt_client = Initializer.startup()

    print(f"{Colors.MAGENTA}\nSystem:\n{Colors.GRAY}{str(llm_client.history)}")

    while(True):
        # Get prompt from user with STT or text input
        print(f"{Colors.GREEN}\nUser:{Colors.RESET}")
        user_input = ""
        if stt_client:
            stt_text_stream = stt_client.prompt()
            for text in stt_text_stream:
                if text is not None:
                    user_input = text
                    print(f"{text}                    ", end="\r")
                else:
                    break
            print()
        else:
            user_input = input()
            
        # Clear playing audio if speech is enabled
        if tts_client:
            tts_client.play_queue.queue.clear()

        # Stream of tokens returned from LLM
        llm_text_stream = llm_client.prompt({"role": "user", "content": user_input})

        generate_output(llm_client, llm_text_stream, tts_client)


# Generate assistant output
def generate_output(llm_client, llm_text_stream, tts_client):
        print(f"{Colors.RED}\nVega:{Colors.RESET}")
        full_reply = ""
        segment = ""
        for text in llm_text_stream:
            sys.stdout.write(text)
            sys.stdout.flush()
            
            segment += text
            full_reply += text

            if tts_client and len(segment) > 20 and re.fullmatch(r'[^A-Z0-9]{2}[.?!:\n]', segment[-3:], re.DOTALL):
                tts_client.queue_tts(segment.strip())
                segment = ""
        if tts_client and segment != "":
            tts_client.queue_tts(segment.strip())
        print("")
        
        # Add finished reply to message history
        llm_client.history.append({"role": "assistant", "content": full_reply.strip()})

        # Recursively prompts itself with the function output until there is no function call
        if "<function_call>" in full_reply:
            prompt = ""
            print(f"{Colors.BLUE}\nFunction Response:{Colors.GRAY}")
            for output_dict in parse_and_call(full_reply):
                output_str = f"<function_response>{str(output_dict)}</function_response>"
                print(output_str)
                prompt += output_str
            llm_text_stream = llm_client.prompt({"role": "function", "content": prompt})
            generate_output(llm_client, llm_text_stream, tts_client)


# Calls functions found in assistant response
def parse_and_call(full_reply):
    function_calls = re.findall(r"<function_call>.*?</function_call>", full_reply)

    for function_call in function_calls:
        json_string = function_call.lstrip("<function_call>").rstrip("</function_call>")
        try:
            loaded_json = json.loads(json_string)
            function_name = loaded_json.get("name")
            function_to_call = getattr(functions, function_name, None)
            function_args = loaded_json.get("arguments", {})

            function_response = function_to_call(*function_args.values())
            results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        except Exception as e:
            yield e
            continue
        yield results_dict


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    MAGENTA = '\033[95m'


if __name__ == "__main__":
    main()