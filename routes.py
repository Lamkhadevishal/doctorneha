from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from core.ragpipeline import RAGPipeline


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  


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


@app.route('/api/chat', methods=['POST'])
def chat_api():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
       
        response = rag.ask(question=user_message, session_id=session['username'])
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500
    

@app.route('/ping')
def ping():
    return "pong", 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))  # Render provides this dynamically
    app.run(host='0.0.0.0', port=port)
