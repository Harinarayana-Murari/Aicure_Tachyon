import sys
import os
import pandas as pd
import joblib

def predict_heart_rate(file_address, model_folder='models', model_filename='gradient_boosting_model.joblib', output_file='results.csv'):
    # Extract the file name from the file address
    input_file = os.path.basename(file_address)

    # Construct the full path to the trained model
    model_path = os.path.join(model_folder, model_filename)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    # Load the input data
    data = pd.read_csv(file_address)

    # Load the trained model
    loaded_model = joblib.load(model_path)

    # Extract features for prediction
    features = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT',
                'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'SD1', 'SD2', 'sampen', 'higuci',
                'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD',
                'pNN25', 'pNN50', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR',
                'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
                'KURT_REL_RR', 'SKEW_REL_RR']

    x_data = data[features]
    x_uuid = data['uuid']

    # Make predictions on the test set using the loaded model
    y_pred_loaded = loaded_model.predict(x_data)

    # Add predicted heart rates to the dataframe
    data['HR'] = pd.DataFrame(y_pred_loaded)

    # Create a new dataframe with 'uuid' and 'HR' columns
    result_df = pd.concat([x_uuid, data['HR']], axis=1)

    # Drop rows with NaN values
    result_df = result_df.dropna()

    # Save the results to a CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <file_address>")
        sys.exit(1)

    file_address = sys.argv[1]
    predict_heart_rate(file_address)
