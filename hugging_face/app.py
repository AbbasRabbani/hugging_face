#from flask import Flask, jsonify, render_template, request
'''
### from fastapi import FastAPI, HTTPException, Request
"""from pydantic import BaseModel

from model import get_response

#app = Flask(__name__)

app = FastAPI()

#### "" @app.route('/')
" def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True) """


"class Message(BaseModel):
# message: str
""
"@app.get("/")
"async def read_root():
   # return {"message": "Welcome to the Simple Chatbot API"} 
""""
@app.post("/predict")
async def predict(message: Message):
    response = get_response(message.message)
    return {"response": response} "

    '''