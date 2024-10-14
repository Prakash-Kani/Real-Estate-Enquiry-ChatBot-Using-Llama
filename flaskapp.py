from flask import Flask, request, jsonify
from Real_Estate_Enquiry import Enquiry_Chain

from datetime import datetime as dt
import os


app = Flask(__name__)




@app.route('/Enquiry', methods=['POST'])
def chatbot():
    # Get the input data from the request
    data = request.get_json()

    if 'session_id' not in data:
        return jsonify({'error': 'No session_id provided'}), 400
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400


    prompt = data['prompt']
    session_id = data['session_id']


    
    if prompt  and session_id:
        print(prompt)
        enquiry_chain = Enquiry_Chain(r'Databases\66f666f956b4d90686543fc9')
        result= enquiry_chain.invoke({"input": prompt},
                                                    config={"configurable": {"session_id": session_id}})["answer"]
    
        

        # response = {'report': result, 'time_stamp': dt.now()}
        response = {'response': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': ' Error Occured'}), 400

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)
