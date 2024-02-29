from flask import Flask, render_template, request, jsonify
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import joblib
import shap
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os
from pyngrok import ngrok

app = Flask(__name__)

# For OpenAI
load_dotenv()
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
)


def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def promt(text, algo_name, class_name):
    return f"""
    Provide a detailed description of the outcome obtained using the {algo_name} algorithm, enclosed within triple quotes. \
    Additionally, explain the reason behind the model's output being classified as an {class_name} based on the {algo_name} result. \
    Present the information in HTML and bootstrap css format, just give inside of <div></div> parts. \

    Text:
    ```{text}```
    """



# Load CSV data as a global variable
csv_data = pd.read_csv('static/KDDTest+.csv')
# Load the LabelEncoder objects
loaded_label_encoders = joblib.load('static/label_encoders.pkl')
# Identify categorical columns
categorical_cols = ['protocol_type', 'service', 'flag', 'outcome']
# Load the trained model from the file
loaded_model = joblib.load('static/random_forest_model.joblib')
class_names=["Anomaly","Normal"]


X_data= csv_data.copy()


# Transform the test data using loaded LabelEncoders
for col in categorical_cols:
    X_data[col] = loaded_label_encoders[col].transform(X_data[col])

X_data = X_data.drop('outcome', axis=1)
csv_data = csv_data.drop('outcome', axis=1)
# Create a LIME Tabular Explainer
explainer = LimeTabularExplainer(training_data=X_data.values, feature_names=X_data.columns, mode="classification", class_names=class_names)

# Create a SHAP explainer for the model
explainer_shap = shap.TreeExplainer(loaded_model)  # Use TreeExplainer for tree-based models

    

@app.route('/')
def index():
    return render_template('index.html', data=csv_data.values, columns_name=csv_data.columns)

@app.route('/update_data', methods=['POST'])
def update_data():
    global csv_data
    
    selected_row_index = int(request.form['selected_row_index'])
    selected_row_data = csv_data.iloc[selected_row_index].to_dict()

    # Modify this part to do something with the selected row data
    #print(selected_row_data)
    
    
    # Select an instance from the test set that you want to explain
    instance_to_explain = X_data.iloc[selected_row_index].values

    # Use the explainer to explain the prediction for this instance
    explanation = explainer.explain_instance(instance_to_explain, loaded_model.predict_proba,  num_features=len(X_data.columns.values))
    lime_values = explanation.as_list()
    #print(lime_values)

    # Render the explanation as HTML
    explanation_html = explanation.as_html()
    explanation_html = explanation_html.replace(
    '.lime.table_div',
    '.lime.table_div {flex: 1 0 300px;}')
    


    

    shap_values = explainer_shap.shap_values(X_data.iloc[[selected_row_index]])
    predicted_class = loaded_model.predict([instance_to_explain])[0]
    # visualize the summary plot
    plt.clf()
    shap.summary_plot(shap_values[predicted_class], X_data.iloc[[selected_row_index]], show=False, max_display=len(X_data.columns.values))
    shap_plot_url = "static/Images/shap_plot_"+str(selected_row_index)+".png"  # Save the SHAP plot to a file
    plt.savefig(shap_plot_url)
    plt.close()


    class_name_text=class_names[predicted_class]
    # shap_values_sample = shap_values[predicted_class]
    shap_values_sample = pd.DataFrame(data=shap_values[predicted_class], columns=X_data.columns)
    

    #print("Predicted class:", class_name_text)
    #print("SHAP values for the predicted class:", shap_values_sample)
    
    

    
    
    response_lime_openai = get_completion(promt(lime_values, "Lime", class_name_text))
    response_shap_openai = get_completion(promt(shap_values_sample, "SHAP", class_name_text))
    #response_lime_openai="test lime"
    #response_shap_openai="test shap"
    
    
    # Create a JSON response
    response = {
        "class_names" : class_name_text,
        "selected_row_data" : selected_row_data,
        "explanation_html" : explanation_html,
        "response_lime_openai" : response_lime_openai,
        "response_shap_openai" : response_shap_openai,
        "shap_img" : shap_plot_url
    }

        
    return jsonify(response)

if __name__ == '__main__':
    # Start ngrok when the app is run
    ngrok_tunnel = ngrok.connect(5000)
    print(" * Running on", ngrok_tunnel.public_url)

    # Update the app to use the ngrok tunnel address
    app.run(port=5000)
