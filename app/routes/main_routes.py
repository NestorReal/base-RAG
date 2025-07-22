from flask import Blueprint, request, render_template, jsonify, current_app # Ensure current_app is imported
from werkzeug.utils import secure_filename
import os
from app import db
from app.models import Document
from app.utils.text_extraction import extract_text_from_pdf, extract_text_from_word
from app.utils.vectorization import chunk_text, vectorize_text

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        # Use current_app.config instead of bp.config as corrected previously
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_ext = filename.split('.')[-1].lower()
        text_content = "" # Using a distinct variable name for clarity
        if file_ext == 'pdf':
            text_content = extract_text_from_pdf(file_path)
        elif file_ext in ['doc', 'docx']:
            text_content = extract_text_from_word(file_path)
        else:
            os.remove(file_path) # Clean up temporary file
            return jsonify({"error": "Unsupported file format"}), 400
        
        # --- Add this check for empty content after extraction ---
        if not text_content or not text_content.strip():
            os.remove(file_path)
            return jsonify({"error": "Document is empty or no readable text could be extracted"}), 400
        # --------------------------------------------------------

        chunks = chunk_text(text_content) # This now returns a list of strings
        
        # --- Handle cases where no meaningful chunks are generated ---
        if not chunks:
            os.remove(file_path)
            return jsonify({"error": "Document could not be broken into meaningful chunks for vectorization"}), 400
        # -------------------------------------------------------------

        vectors = [vectorize_text(chunk) for chunk in chunks] # Now 'chunk' will be a string, as expected by vectorize_text

        # --- Handle cases where no vectors could be generated ---
        if not vectors:
            os.remove(file_path)
            return jsonify({"error": "No vectors could be generated from document chunks"}), 500
        # --------------------------------------------------------

        # Calculate combined_vector (assuming all vectors are of the same length)
        # This will work as long as 'vectors' is not empty, which is now handled above
        combined_vector = [sum(x)/len(x) for x in zip(*vectors)]

        # Assuming you want to store the full text as 'text' and the combined_vector as 'vector'
        new_document = Document(text=text_content, vector=combined_vector)
        db.session.add(new_document)
        db.session.commit()

        os.remove(file_path)  # Eliminar el archivo temporal

        return jsonify({"message": "Document uploaded and processed successfully!", "document_id": new_document.id}), 200
    
    return jsonify({"error": "File upload failed"}), 400 # Return JSON for generic failure

@bp.route('/delete_documents', methods=['POST'])
def delete_documents():
    try:
        num_deleted = db.session.query(Document).delete()
        db.session.commit()
        return jsonify({"message": f"All {num_deleted} documents deleted successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@bp.route('/test_db')
def test_db():
    try:
        num_documents = Document.query.count() # Using count() is more efficient
        return jsonify({"message": f"Successfully connected to the database. Number of documents: {num_documents}"}), 200
    except Exception as e:
        return jsonify({"message": f"Database connection failed: {str(e)}"}), 500