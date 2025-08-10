from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from core.ragpipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!

# Dummy user store
USER_DATA = {
    'Dr neha': 'neha123',
}
rag = RAGPipeline(doc_path='data/doctorneha.pdf')

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('chatbot'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USER_DATA and USER_DATA[username] == password:
            session['username'] = username
            return redirect(url_for('chatbot'))
        else:
            error = "Invalid username or password"
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template('chatbot.html', username=session['username'])

# Example API endpoint for chatbot response (replace with your RAG logic)
@app.route('/api/chat', methods=['POST'])
def chat_api():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Pass the current username as session_id for personalized chat history
        response = rag.ask(question=user_message, session_id=session['username'])
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)
