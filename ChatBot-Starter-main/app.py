from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session usage

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize a global dictionary to store chat history for each session
chat_histories = {}

@app.route("/")
def index():
    session['session_id'] = session.get('session_id', str(len(chat_histories)))
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    session_id = session['session_id']

    # Retrieve or initialize chat history
    chat_history_ids = chat_histories.get(session_id)

    # Tokenize user input and append
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Save updated chat history
    chat_histories[session_id] = chat_history_ids

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    app.run(debug=True)
