from flask import Blueprint, request, jsonify
from app import db
from app.models import Document
from app.utils.vectorization import vectorize_text
from app.utils.similarity import cosine_similarity
import openai
from transformers import BertModel, BertTokenizer

bp = Blueprint('documents', __name__)

@bp.route('/documents', methods=['GET'])
def get_documents():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    documents = Document.query.paginate(page=page, per_page=per_page)
    results = [
        {
            "id": doc.id,
            "text": doc.text,
            "vector": doc.vector
        } for doc in documents.items]
    
    return jsonify({
        'documents': results,
        'total': documents.total,
        'pages': documents.pages,
        'current_page': documents.page
    })

@bp.route('/query', methods=['POST'])
def query():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    try:
        query_vector = vectorize_text(user_query)

        documents = Document.query.all()

        similarities = [
            (doc.id, doc.text, cosine_similarity(query_vector, doc.vector))
            for doc in documents
        ]

        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        top_k = 3
        top_fragments = [sim[1] for sim in similarities[:top_k]]
        context = "\n".join(top_fragments)

        max_context_tokens = 2000
        context_tokens = tokenizer(context, return_tensors='pt')['input_ids'][0]

        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        num_responses = 3
        responses = []

        for _ in range(num_responses):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": f"Contexto: {context}"},
                    {"role": "user", "content": f"Pregunta: {user_query}"}
                ],
                max_tokens=100
            )
            answer = response.choices[0].message['content'].strip()
            responses.append(answer)

        return jsonify({
            "responses": responses,
            "query": user_query,
            "context": context,
            'total_documents': len(documents),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
