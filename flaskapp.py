from flask import Flask, request, jsonify
from Real_Estate_Enquiry2 import Enquiry_Chain
from property_data_loader import property_data_ingest
from property_listing_chain import create_chain
from property_listing_chain1 import create_chain_
from websearch_property_listing import run_create_chain
# from websearch_ import rusult_create_chain
from new import result_chain


from datetime import datetime as dt
import os

# api_key = "Enter your api key here"
# cse_id = "Enter your CSE id here"
app = Flask(__name__)


UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


DB_FOLDER = 'Databases'
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)
app.config['DB_FOLDER'] = DB_FOLDER


@app.route('/Document-Upload', methods=['POST'])
def ingest_pdf():
    # Ensure that 'filename' and 'pdf' are part of the form data
    if 'filename' not in request.form :
        return jsonify({'error': 'Filename is missing'}), 400
    if  'csv' not in request.files:
        return jsonify({'error': ' CSV file is missing'}), 400
    # Get the filename and PDF file from the request
    filename = request.form['filename']
    csv_file = request.files['csv']

    # Check if the file is a valid PDF
    if csv_file.filename == '' or not csv_file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV.'}), 400

    # Save the PDF file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    csv_file.save(file_path)

    # if topic:
    #     text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.txt')
    #     with open(text_file_path, 'w') as text_file:
    #         text_file.write(history)


    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)
    print(file_path, persist_directory)
    property_data_ingest(file_path=file_path, persist_directory = persist_directory)
    # Return success response
    return jsonify({'message': 'File uploaded and processed successfully'}), 200




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
    


@app.route('/Find-Property', methods=['POST'])
def finding_properties():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'city' not in data:
        return jsonify({'error': 'No city provided'}), 400
    if 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    city = data['city']
    area = None
    square_ft = None
    price = None
    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)

    if 'area' in data:
        area = data['area']
    if 'square_ft' in data:
        square_ft = data['square_ft']
    if 'price' in data:
        price = data['price']
    


    if filename and city and area and square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price} in {city}, specifically in {area}."
    elif filename and city and area and price:
        prompt = f"I'm searching for plots with a price range of {price} in {city}, specifically in {area}."
    elif filename and city and area and square_ft:
        prompt = f"I'm searching for plots over {square_ft}  square feet in {city}, specifically in {area}."
    elif filename and city and area:
        prompt = f"I'm searching for plots  in {city}, specifically in {area}."
    elif filename and square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price}."
    
    elif filename and city:
        prompt = f"I'm searching for plots  in {city}."
    elif filename and area:
        prompt = f"I'm searching for plots  in {area}."


    if prompt and persist_directory:
        print(prompt)
        # print(persist_directory)

        question_generation_chain = create_chain(persist_directory)
        result= question_generation_chain.invoke(prompt)
    
        # result = chain.invoke("generate the report")

        # response = {'report': result, 'time_stamp': dt.now()}
        response = {'Property': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400


@app.route('/LLM-Property', methods=['POST'])
def finding_properties_():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'city' not in data:
        return jsonify({'error': 'No city provided'}), 400
    
    
    
    city = data['city']
    area = None
    square_ft = None
    price = None

    if 'area' in data:
        area = data['area']
    if 'square_ft' in data:
        square_ft = data['square_ft']
    if 'price' in data:
        price = data['price']
    


    if city and area and square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price} in {city}, specifically in {area}."
    elif city and area and price:
        prompt = f"I'm searching for plots with a price range of {price} in {city}, specifically in {area}."
    elif city and area and square_ft:
        prompt = f"I'm searching for plots over {square_ft}  square feet in {city}, specifically in {area}."
    elif city and area:
        prompt = f"I'm searching for plots  in {city}, specifically in {area}."
    elif square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price}."
    
    elif city:
        prompt = f"I'm searching for plots  in {city}."
    elif area:
        prompt = f"I'm searching for plots  in {area}."


    if prompt:
        print(prompt)
        # print(persist_directory)

        question_generation_chain = create_chain_()
        result= question_generation_chain.invoke(prompt)
    
        # result = chain.invoke("generate the report")

        # response = {'report': result, 'time_stamp': dt.now()}
        response = {'Property': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400



@app.route('/Web-run', methods=['POST'])
def web_properties_():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'city' not in data:
        return jsonify({'error': 'No city provided'}), 400
    
    
    
    city = data['city']
    area = None
    square_ft = None
    price = None

    if 'area' in data:
        area = data['area']
    if 'square_ft' in data:
        square_ft = data['square_ft']
    if 'price' in data:
        price = data['price']
    


    if city and area and square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price} in {city}, specifically in {area}."
    elif city and area and price:
        prompt = f"I'm searching for plots with a price range of {price} in {city}, specifically in {area}."
    elif city and area and square_ft:
        prompt = f"I'm searching for plots over {square_ft}  square feet in {city}, specifically in {area}."
    elif city and area:
        prompt = f"I'm searching for plots  in {city}, specifically in {area}."
    elif square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price}."
    
    elif city:
        prompt = f"I'm searching for plots  in {city}."
    elif area:
        prompt = f"I'm searching for plots  in {area}."


    if prompt:
        print(prompt)
        # print(persist_directory)

        question_generation_chain = run_create_chain(api_key, cse_id)
        result= question_generation_chain.invoke(prompt)
    
        # result = chain.invoke("generate the report")

        # response = {'report': result, 'time_stamp': dt.now()}
        response = {'Property': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400
    
@app.route('/Web-result', methods=['POST'])
def web_result_properties_():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'city' not in data:
        return jsonify({'error': 'No city provided'}), 400
    
    
    
    city = data['city']
    area = None
    square_ft = None
    price = None

    if 'area' in data:
        area = data['area']
    if 'square_ft' in data:
        square_ft = data['square_ft']
    if 'price' in data:
        price = data['price']
    


    if city and area and square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price} in {city}, specifically in {area}."
    elif city and area and price:
        prompt = f"I'm searching for plots with a price range of {price} in {city}, specifically in {area}."
    elif city and area and square_ft:
        prompt = f"I'm searching for plots over {square_ft}  square feet in {city}, specifically in {area}."
    elif city and area:
        prompt = f"I'm searching for plots  in {city}, specifically in {area}."
    elif square_ft and price:
        prompt = f"I'm searching for plots over {square_ft}  square feet with a price range of {price}."
    
    elif city:
        prompt = f"I'm searching for plots  in {city}."
    elif area:
        prompt = f"I'm searching for plots  in {area}."


    if prompt:
        print(prompt)
        # print(persist_directory)

        question_generation_chain = result_chain()
        result= question_generation_chain.invoke(prompt)
    
        # result = chain.invoke("generate the report")

        # response = {'report': result, 'time_stamp': dt.now()}
        response = {'Property': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8501, debug=True)