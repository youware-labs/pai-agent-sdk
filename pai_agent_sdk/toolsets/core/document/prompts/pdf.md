<pdf-convert-guidelines>

<best-practices>
- For large PDFs, convert in chunks using page_start/page_end to avoid memory issues
- Use page_end=-1 only when you need the entire document
- Check total_pages in response to plan subsequent conversion calls
- The exported markdown and images are in `export_{filename}/` next to the source file
</best-practices>

</pdf-convert-guidelines>
