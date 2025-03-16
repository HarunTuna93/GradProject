import os
import openai

openai.api_key = "xx"

input_folder = r"xx"
output_folder = r"xx"

model_name = "gpt-4o-mini"

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

        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Türkçe metin düzenleyici."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16000
            )

            improved_text = response.choices[0].message.content.strip()

            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(improved_text)

            print(f"Processed and improved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
