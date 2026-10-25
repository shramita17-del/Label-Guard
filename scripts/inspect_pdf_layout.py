import pypdf

reader = pypdf.PdfReader('reports/generated/LabelGuard_Report_f090b8a1_redesigned.pdf')
print(f"Total pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    print(f"\n--- PAGE {i+1} ---")
    print(f"Images: {len(page.images)}")
    text = page.extract_text()
    for line in text.splitlines():
        if line.strip():
            print(f"  {line.strip()[:80]}")
