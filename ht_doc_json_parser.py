import json

input_path = r"xx"
output_path = r"xx"

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    analyze_result = data.get("analyzeResult", {})
    pages = analyze_result.get("pages", [])
    all_lines = []
    for page in pages:
        for line_item in page.get("lines", []):
            text_content = line_item.get("content", "").strip()
            if text_content.isdigit():
                continue
            all_lines.append(text_content)
    full_text = "\n".join(all_lines)
    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(full_text)

    print(f"Extraction completed! Check '{output_path}' for results.")

if __name__ == "__main__":
    main()
