import document_generator


def test_document_definitions_have_required_fields():
    assert document_generator.DOCUMENTS

    for document in document_generator.DOCUMENTS:
        assert {"title", "type", "client_industry", "sections"} <= document.keys()
        assert document["sections"]


def test_generate_all_documents_creates_expected_pdfs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(document_generator, "DATA_DIR", tmp_path)

    document_generator.generate_all_documents()

    generated_files = sorted(tmp_path.glob("*.pdf"))

    assert len(generated_files) == len(document_generator.DOCUMENTS)
    assert all(file.stat().st_size > 0 for file in generated_files)
    assert all("/" not in file.name for file in generated_files)

    output = capsys.readouterr().out
    assert f"All {len(document_generator.DOCUMENTS)} documents saved to:" in output


def test_pdf_document_can_render_single_document(tmp_path):
    sample = document_generator.DOCUMENTS[0]
    pdf = document_generator.PDFDocument(
        sample["title"],
        sample["type"],
        sample["client_industry"],
    )
    pdf.alias_nb_pages()
    pdf.add_title_page()
    pdf.add_page()

    for section_title, section_content in sample["sections"].items():
        pdf.add_section(section_title, section_content)

    output_path = tmp_path / "single_document.pdf"
    pdf.output(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert pdf.page_no() >= 2
