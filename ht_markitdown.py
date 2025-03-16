from markitdown import MarkItDown

# Paths
pdf_input = r"x"
md_output = r"x"

# Create a MarkItDown object
md = MarkItDown()

# Convert PDF to Markdown
result = md.convert(pdf_input)

# Write the resulting Markdown text to a file
with open(md_output, "w", encoding="utf-8") as f:
    f.write(result.text_content)

print(f"Markdown saved to: {md_output}")
