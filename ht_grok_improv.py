import os
import requests

def chat_with_xai_api(api_url, api_key, user_messages):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        "messages": user_messages,
        "model": "grok-2-1212",
        "stream": False,
        "temperature": 0
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        json_response = response.json()
        return json_response.get("choices", [{}])[0].get("message", {}).get("content", "No content in response.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    API_URL = "https://api.x.ai/v1/chat/completions"
    API_KEY = "xx"

    input_folder = r"x"
    output_folder = r"x"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            with open(input_file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
                print(f"Read {len(original_text)} characters from {filename}")

            prompt = (
                "Bu Türkçe metini kesinlikle anlam kaybı olmadan dilbilgisi ve yazım olarak düzelt. "
                "daha anlaşılır hale getir. Metnin özgün anlamını korumaya çok özen göster. "
                "\"--- Sayfa 7 ---\" şeklindeki sayfa bilgilerini olduğu gibi koru. "
                "Bana düzeltilmiş metin dışında cevap verme.\n\n"
                f"{original_text}"
            )

            messages = [
                {"role": "system", "content": "Türkçe metin düzenleyici."},
                {"role": "user", "content": prompt}
            ]

            try:
                improved_text = chat_with_xai_api(API_URL, API_KEY, messages)

                output_file_path = os.path.join(output_folder, filename)
                with open(output_file_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(improved_text)

                print(f"Processed and improved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
