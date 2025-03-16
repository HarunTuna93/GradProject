import os
import re


def chunk_text_by_sentence(input_file, output_folder, max_words=850):
    os.makedirs(output_folder, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if current_word_count + words_in_sentence <= max_words:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    for i, chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(output_folder, f'chunk_{i}.txt')
        with open(chunk_filename, 'w', encoding='utf-8') as f:
            f.write(chunk)

    print(f"Created {len(chunks)} chunk(s) in '{output_folder}'.")


if __name__ == '__main__':
    input_file_path = r"xx"
    output_folder_path = r"xx"

    chunk_text_by_sentence(input_file_path, output_folder_path, max_words=850)
