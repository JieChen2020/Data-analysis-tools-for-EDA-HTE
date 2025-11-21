import functools
import os
import shlex
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.neural_network import MLPClassifier
import seaborn as sns

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
import langsmith
import random
from langchain.prompts import PromptTemplate

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_DEBUG"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "xx"
os.environ["LANGCHAIN_API_KEY"] = "xx"
os.environ["LANGCHAIN_PROJECT"] = "xx"


def param_decorator(func):
    @functools.wraps(func)
    def wrapper(param_string):
        params = {}
        args = []
        if ',' in param_string:
            param_split = param_string.split(',')
        elif ' ' in param_string:
            param_split = param_string.split(' ')
        else:
            param_split = param_string.strip()
        for param in param_split:
            param = shlex.split(param)[0]
            if "=" in param:
                key, value = param.split('=')
            elif ":" in param:
                key, value = param.split(':')
            else:
                args.append(param.strip())
                continue
            params[key.strip()] = value.strip()
        return func(*args, **params)

    return wrapper


@param_decorator
def Generate_Morganfingerprints_from_csv(csv_name: str, radius: str, size: str):
    """
    Generate morgan fingerprints for the SMILES strings in a CSV file and save to a new CSV file.

    Args:
        csv_name: file name of the input CSV containing SMILES strings.

    Returns:
        str: Message indicating the completion of processing and the path to the output CSV file.
    
    Note: Your input should ideally be in the form of something like 'csv_name=filename.csv, radius=3, size=512'

    """
    global RADIUS, SIZE

    RADIUS = int(radius)
    SIZE = int(size)
    # print(f"\nRADIUS:{RADIUS}\nSIZE:{SIZE}")
    test_data_input_path = "D:/PythonProject/Chem_AI/demo_test.csv"
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    # Read the input CSV file
    data = pd.read_csv(test_data_input_path)

    # Ensure there are at least three columns for SMILES strings
    if len(data.columns) < 3:
        return "The input CSV file must have at least three columns for SMILES strings."

    # Process each SMILES string and generate fingerprints
    fingerprints = []
    for idx, row in data.iterrows():
        row_fps = []
        for smi in row[:3]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Generate Morgan fingerprints with radius 3 and 512 bits
                generator = GetMorganGenerator(radius=RADIUS, fpSize=SIZE)
                morgan_fp = generator.GetFingerprint(mol)
                # Convert the fingerprint to a binary string representation
                binary_string = morgan_fp.ToBitString()
                row_fps.append(binary_string)
            else:
                # In case the SMILES string cannot be converted to a molecule
                row_fps.append("Invalid SMILES")
        fingerprints.append(row_fps)

    # Create a new DataFrame with the fingerprints and the remaining data
    fp_data = pd.DataFrame(fingerprints, columns=['B_Fingerprint', 'C_Fingerprint', 'Product_Fingerprint'])
    result_data = pd.concat([fp_data, data.iloc[:, 3:]], axis=1)
    # print(fingerprints, result_data)

    # Save the resulting DataFrame to a new CSV file
    result_data.to_csv(test_data_output_path, index=False)

    return f"\nSMILES strings in the {csv_name} file have been processed, and the morgan fingerprints features are saved to the path {test_data_output_path}\n"


@param_decorator
def Generate_Simplified_Morganfingerprints_from_csv(csv_name: str, radius: str, size: str):
    """
    Generate Simplified morgan fingerprints for the SMILES strings in a CSV file and save to a new CSV file.

    Args:
        csv_name: file name of the input CSV containing SMILES strings.

    Returns:
        str: Message indicating the completion of processing and the path to the output CSV file.

    Note: Your input should ideally be in the form of something like 'csv_name=filename.csv, radius=3, size=512'

    """
    global RADIUS, SIZE

    RADIUS = int(radius)
    SIZE = int(size)
    # print(f"\nRADIUS:{RADIUS}\nSIZE:{SIZE}")
    test_data_input_path = "D:/PythonProject/Chem_AI/simpilified_demo_test.csv"
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    # Read the input CSV file
    data = pd.read_csv(test_data_input_path)

    # Ensure there are at least three columns for SMILES strings
    if len(data.columns) < 3:
        return "The input CSV file must have at least three columns for SMILES strings."

    # Process each SMILES string and generate fingerprints
    fingerprints = []
    for idx, row in data.iterrows():
        row_fps = []
        for smi in row[:3]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Generate Morgan fingerprints with radius 3 and 512 bits
                generator = GetMorganGenerator(radius=RADIUS, fpSize=SIZE)
                morgan_fp = generator.GetFingerprint(mol)
                # Convert the fingerprint to a binary string representation
                binary_string = morgan_fp.ToBitString()
                row_fps.append(binary_string)
            else:
                # In case the SMILES string cannot be converted to a molecule
                row_fps.append("Invalid SMILES")
        fingerprints.append(row_fps)

    # Create a new DataFrame with the fingerprints and the remaining data
    fp_data = pd.DataFrame(fingerprints, columns=['B_Fingerprint', 'C_Fingerprint', 'Product_Fingerprint'])
    result_data = pd.concat([fp_data, data.iloc[:, 3:]], axis=1)

    # Save the resulting DataFrame to a new CSV file
    result_data.to_csv(test_data_output_path, index=False)

    return f"\nSMILES strings in the {csv_name} file have been processed, and the simplified morgan fingerprints features are saved to the path {test_data_output_path}\n"


def Generate_RDKitDescriptors_from_csv(csv_name: str):
    """
    Generate RDKit descriptors for the SMILES strings in a CSV file and save to a new CSV file.
    
    Args:
        csv_name: Path to the input CSV file containing SMILES strings.

    Returns:
        str: Message indicating the completion of processing and the path to the output CSV file.
    """
    test_data_input_path = "D:/PythonProject/Chem_AI/demo_test.csv"
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_descriptors.csv'

    data = pd.read_csv(test_data_input_path)
    if len(data.columns) < 2:
        return "Error: The CSV file must have at least three columns for SMILES strings."

    descriptor_keys = ['MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed',
                       'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
                       'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
                       'MolLogP', 'MolMR', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                       'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                       'NumHDonors', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
                       'RingCount',
                       'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
                       'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN',
                       'fr_Imine',
                       'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                       'fr_Nhpyrrole',
                       'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
                       'fr_amide',
                       'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
                       'fr_benzene',
                       'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide',
                       'fr_ester', 'fr_ether',
                       'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole',
                       'fr_imide',
                       'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
                       'fr_methoxy',
                       'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                       'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                       'fr_phenol_noOrthoHbond',
                       'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide',
                       'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
                       'fr_term_acetylene',
                       'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    descriptors_list = []
    # Process each of the first three SMILES string columns
    for col in data.columns[:2]:  # Assumes the first two columns are SMILES strings
        descriptors = []
        for smi in data[col]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Calculate the descriptors for each molecule
                desc = [Descriptors.__dict__[dk](mol) if dk in Descriptors.__dict__ else None for dk in descriptor_keys]
                descriptors.append(desc)
            else:
                # Append None for each descriptor if the SMILES string is invalid
                descriptors.append([None] * len(descriptor_keys))
        # Append the results to the descriptors_list
        descriptors_list.append(pd.DataFrame(descriptors, columns=[f"{col}_{dk}" for dk in descriptor_keys]))

    # Concatenate all descriptors along the column axis
    all_descriptors = pd.concat(descriptors_list, axis=1)
    result_data = pd.concat([all_descriptors, data.iloc[:, 3:]],
                            axis=1)  # Concatenate descriptors with conversion data columns
    # Save the resulting DataFrame to a new CSV file
    result_data.to_csv(test_data_output_path, index=False)

    return f"\nProcessed SMILES strings and saved descriptors to {test_data_output_path}"


def load_and_prepare_RDKitdescriptors_data(csv_file):
    """
    Load data from a CSV file and prepare it for machine learning models.

    Args:
        csv_file: Path to the CSV file containing the data.

    Returns:
        tuple: A tuple containing the features as a numpy array and the labels as a numpy array.
    """
    data = pd.read_csv(csv_file)
    y_label_index = -2

    # Assuming that all except the last two columns are features if y_label_index is -2 (the second last column as label)
    if y_label_index == -2:
        X = data.iloc[:, :-2]  # All columns except the last two are features
        y = data.iloc[:, y_label_index]  # The second last column is the label
    else:
        # Adjust according to specific layout if needed
        X = data.iloc[:, :y_label_index]  # Features up to the label index
        y = data.iloc[:, y_label_index]  # Label column

    # Convert features to float if not already
    X = X.apply(pd.to_numeric, errors='coerce', axis=1)

    # Handling possible NaNs in features
    X = X.fillna(0.0)

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Convert labels to category based on a specific threshold or logic
    y = y.apply(lambda x: 0 if x <= 33 else (1 if x <= 66 else 2))

    return X_normalized, y


def MLP_Classifier_RDKitDescriptors(csv_name: str):
    """
    The MLP algorithm trained based on the RDKitDescriptors data set is used to predict the data set.

    Args:
        csv_name: The name of the RDKitDescriptors data file used for the prediction model.

    Returns:
        str: A string describing the accuracy of the model on the training set and test set.
    """
    global RADIUS, SIZE
    test_data_output_path = 'D:/PythonProject/Chem_AI/demo_test_descriptors.csv'
    test_data_output_path1 = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        # X, y = load_and_prepare_combined_features(test_data_output_path, test_data_output_path1)
        X, y = load_and_prepare_RDKitdescriptors_data(test_data_output_path)

        model = MLPClassifier(max_iter=800)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def MLP_Classifier_Morganfingerprints(csv_name: str):
    """
    The MLP algorithm trained based on the Morgan fingerprints data set is only used to predict the Morgan fingerprints data set.

    Args:
        csv_name: The name of the Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = MLPClassifier(max_iter=500)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"

        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def RandomForest_Classifier_Morganfingerprints(csv_name: str):
    """
    The RandomForest algorithm trained based on the Morgan fingerprints data set is only used to predict the Morgan fingerprints data set.

    Args:
        csv_name: The name of the Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = RandomForestClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"

        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def AdaBoost_Classifier_Morganfingerprints(csv_name: str):
    """
    The AdaBoost algorithm trained based on the Morgan fingerprints data set is only used to predict the Morgan fingerprints data set.

    Args:
        csv_name: The name of the Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = AdaBoostClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"

        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def KNeighbors_Classifier_Morganfingerprints(csv_name: str):
    """
    The KNeighbors algorithm trained based on the Morgan fingerprints data set is only used to predict the Morgan fingerprints data set.

    Args:
        csv_name: The name of the Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = KNeighborsClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"

        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def load_and_prepare_Morganfingerprints_data(csv_file):
    data = pd.read_csv(csv_file)
    features1 = data.iloc[:, 0].apply(lambda x: np.array([int(i) for i in x]))
    features2 = data.iloc[:, 1].apply(lambda x: np.array([int(i) for i in x]))
    features3 = data.iloc[:, 2].apply(lambda x: np.array([int(i) for i in x]))
    features = features1 + features2 - features3
    features = np.vstack(features)

    labels = data.iloc[:, 3]
    features = np.array(features.tolist())
    labels = labels.apply(lambda x: 0 if x <= 33 else (1 if x <= 66 else 2))
    return features, labels


def load_and_prepare_combined_features(rdkit_csv, morgan_csv):
    """
    Combine RDKit descriptors and Morgan fingerprints into a single feature matrix.

    Args:
        rdkit_csv (str): Path to the CSV file containing RDKit descriptors.
        morgan_csv (str): Path to the CSV file containing Morgan fingerprints.

    Returns:
        tuple: (X_combined, y_combined)
               X_combined -> concatenated feature matrix (numpy.ndarray)
               y_combined -> categorical labels (pandas.Series)
    """

    rdkit_data = pd.read_csv(rdkit_csv)
    y_label_index = -2
    if y_label_index == -2:
        X_rdkit = rdkit_data.iloc[:, :-2]
        y = rdkit_data.iloc[:, y_label_index]
    else:
        X_rdkit = rdkit_data.iloc[:, :y_label_index]
        y = rdkit_data.iloc[:, y_label_index]

    X_rdkit = X_rdkit.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    scaler = StandardScaler()
    X_rdkit = scaler.fit_transform(X_rdkit)

    morgan_data = pd.read_csv(morgan_csv)
    features1 = morgan_data.iloc[:, :1].apply(lambda row: np.array([int(x) for x in ''.join(row.values)]), axis=1)
    features2 = morgan_data.iloc[:, 1:2].apply(lambda row: np.array([int(x) for x in ''.join(row.values)]), axis=1)
    X_morgan = np.vstack([np.concatenate([f1, f2]) for f1, f2 in zip(features1, features2)])

    X_combined = np.concatenate([X_rdkit, X_morgan], axis=1)

    y = y.apply(lambda x: 0 if x <= 33 else (1 if x <= 66 else 2))

    return X_combined, y


def MLP_Classifier_Simplified_Morganfingerprints(csv_name: str):
    """
    The MLP algorithm trained based on the Simplified Morgan fingerprints data set is only used to predict the Simplified Morgan fingerprints data set.

    Args:
        csv_name: The name of the Simplified Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    try:

        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = MLPClassifier(max_iter=500)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)
        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def RandomForest_Classifier_Simplified_Morganfingerprints(csv_name: str):
    """
    The RandomForest algorithm trained based on the Simplified Morgan fingerprints data set is only used to predict the Simplified Morgan fingerprints data set.

    Args:
        csv_name: The name of the Simplified Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    try:

        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = RandomForestClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)
        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def AdaBoost_Classifier_Simplified_Morganfingerprints(csv_name: str):
    """
    The AdaBoost algorithm trained based on the Simplified Morgan fingerprints data set is only used to predict the Simplified Morgan fingerprints data set.

    Args:
        csv_name: The name of the Simplified Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    try:

        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = AdaBoostClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)
        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def KNeighbors_Classifier_Simplified_Morganfingerprints(csv_name: str):
    """
    The KNeighbors algorithm trained based on the Simplified Morgan fingerprints data set is only used to predict the Simplified Morgan fingerprints data set.

    Args:
        csv_name: The name of the Simplified Morgan fingerprint data file used for the prediction model.

    return:
        str: A string describing the accuracy of the model on the training set and test set.

    Note: Your input only needs to contain the file name, without any additional information. For example, your input should be "filename.csv" instead of "csv_name='filename.csv'"

    """
    global RADIUS, SIZE
    test_data_output_path = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'  # Output CSV file name

    try:

        X, y = load_and_prepare_Morganfingerprints_data(test_data_output_path)

        model = KNeighborsClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)
        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def RandomForest_Classifier_RDKitDescriptors(csv_name: str):
    """
    The Random Forest algorithm trained based on the RDKitDescriptors data set is used to predict the data set.

    Args:
        csv_name: The name of the RDKitDescriptors data file used for the prediction model.

    Returns:
        str: A string describing the accuracy of the model on the training set and test set.
    """
    global RADIUS, SIZE
    test_data_output_path = 'D:/PythonProject/Chem_AI/demo_test_descriptors.csv'
    test_data_output_path1 = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        # X, y = load_and_prepare_combined_features(test_data_output_path, test_data_output_path1)
        X, y = load_and_prepare_RDKitdescriptors_data(test_data_output_path)

        model = RandomForestClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def AdaBoost_Classifier_RDKitDescriptors(csv_name: str):
    """
    The AdaBoost algorithm trained based on the RDKitDescriptors data set is used to predict the data set.

    Args:
        csv_name: The name of the RDKitDescriptors data file used for the prediction model.

    Returns:
        str: A string describing the accuracy of the model on the training set and test set.
    """
    global RADIUS, SIZE
    test_data_output_path = 'D:/PythonProject/Chem_AI/demo_test_descriptors.csv'
    test_data_output_path1 = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        # X, y = load_and_prepare_combined_features(test_data_output_path, test_data_output_path1)
        X, y = load_and_prepare_RDKitdescriptors_data(test_data_output_path)

        model = AdaBoostClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


def KNeighbors_Classifier_RDKitDescriptors(csv_name: str):
    """
    The KNeighbors algorithm trained based on the RDKitDescriptors data set is used to predict the data set.

    Args:
        csv_name: The name of the RDKitDescriptors data file used for the prediction model.

    Returns:
        str: A string describing the accuracy of the model on the training set and test set.
    """
    global RADIUS, SIZE
    test_data_output_path = 'D:/PythonProject/Chem_AI/demo_test_descriptors.csv'
    test_data_output_path1 = f'D:/PythonProject/Chem_AI/demo_test_simpilified_morgan_fingerprints_{RADIUS}_{SIZE}.csv'

    try:
        # X, y = load_and_prepare_combined_features(test_data_output_path, test_data_output_path1)
        X, y = load_and_prepare_RDKitdescriptors_data(test_data_output_path)

        model = KNeighborsClassifier()
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kfold)

        res = f"\nAverage Accuracy: {results.mean()}\n"
        return res
    except Exception as e:
        return f"Error during model training or evaluation: {str(e)}"


EXTRA_TOOL_NAME_DICT = {
    "GenerateMorganfingerprintsFromCSV": Generate_Morganfingerprints_from_csv,
    "GenerateSimplifiedMorganfingerprintsFromCSV": Generate_Simplified_Morganfingerprints_from_csv,
    "GenerateRDKitDescriptorsFromCSV": Generate_RDKitDescriptors_from_csv,

    "MLPClassifierMorgan": MLP_Classifier_Morganfingerprints,
    "RandomForestClassifierMorgan": RandomForest_Classifier_Morganfingerprints,
    "AdaBoostClassifierMorgan": AdaBoost_Classifier_Morganfingerprints,
    "KNeighborsClassifierMorgan": KNeighbors_Classifier_Morganfingerprints,
    "MLPClassifierSimplifiedMorgan": MLP_Classifier_Simplified_Morganfingerprints,
    "RandomForestClassifierSimplifiedMorgan": RandomForest_Classifier_Simplified_Morganfingerprints,
    "AdaBoostClassifierSimplifiedMorgan": AdaBoost_Classifier_Simplified_Morganfingerprints,
    "KNeighborsClassifierSimplifiedMorgan": KNeighbors_Classifier_Simplified_Morganfingerprints,
    "MLPClassifierRDKit": MLP_Classifier_RDKitDescriptors,
    "RandomForestClassifierRDKit": RandomForest_Classifier_RDKitDescriptors,
    "AdaBoostClassifierRDKit": AdaBoost_Classifier_RDKitDescriptors,
    "KNeighborsClassifierRDKit": KNeighbors_Classifier_RDKitDescriptors,
    # add tools
}

# add tools
TOOLS_MAPPING = {
    "ChemicalTools": [
        "GenerateMorganfingerprintsFromCSV",
        "GenerateSimplifiedMorganfingerprintsFromCSV",
        "GenerateRDKitDescriptorsFromCSV",

        "MLPClassifierMorgan",
        "RandomForestClassifierMorgan",
        "AdaBoostClassifierMorgan",
        "KNeighborsClassifierMorgan",
        "MLPClassifierSimplifiedMorgan",
        "RandomForestClassifierSimplifiedMorgan",
        "AdaBoostClassifierSimplifiedMorgan",
        "KNeighborsClassifierSimplifiedMorgan",
        "MLPClassifierRDKit",
        "RandomForestClassifierRDKit",
        "AdaBoostClassifierRDKit",
        "KNeighborsClassifierRDKit"
    ]
}


def load_tool_from_name(names, llm):
    tools = []
    for name in names:
        if name in EXTRA_TOOL_NAME_DICT:
            func = EXTRA_TOOL_NAME_DICT[name]
            description = func.__doc__ or ""
            tools.append(Tool.from_function(func=func, name=name, description=description, return_direct=False))
            # tools.append(Tool.from_function(func=func, name=name, description=description, return_direct=True))
        else:
            tools.extend(load_tools([name], llm=llm))
    return tools


import langchain_openai.chat_models.base as base


def safe_create_usage_metadata(token_usage):
    input_tokens = token_usage.get("prompt_tokens") or 0
    output_tokens = token_usage.get("completion_tokens") or 0
    total_tokens = token_usage.get("total_tokens") or (input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


base._create_usage_metadata = safe_create_usage_metadata


def get_tool_list(tools_param: List[str]) -> List[str]:
    tool_list = []
    for category in tools_param:
        tool_list.extend(TOOLS_MAPPING.get(category, []))
    return tool_list


def load_tool_from_name(names, llm):
    tools = []
    for name in names:
        if name in EXTRA_TOOL_NAME_DICT:
            func = EXTRA_TOOL_NAME_DICT[name]
            description = func.__doc__ or ""
            tools.append(Tool.from_function(func=func, name=name, description=description, return_direct=False))
        else:
            tools.extend(load_tools([name], llm=llm))
    return tools


def run_langchain(prompt, tools):
    llm = ChatOpenAI(
        model="xx",
        api_key="xx",
        base_url="xx",
        streaming=False,
        request_timeout=180,
        max_retries=0
    )

    tool_list = get_tool_list(tools)
    if tool_list:
        tools = load_tool_from_name(tool_list, llm)


        react_prompt = PromptTemplate.from_template(""" 
You are a reasoning agent that can use tools to solve problems.

You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: you should always think about what to do next
Action: the action to take, one of [{tool_names}]
Action Input: the input to that action
Observation: the result of the action
... (this Thought/Action/Observation can repeat as many times as needed) ...
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
""")

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=react_prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=100
        )

        result = agent_executor.invoke({"input": prompt})
        print(result["output"] if "output" in result else result)

    else:
        response = llm.invoke(prompt)
        print(response)


if __name__ == "__main__":
    global RADIUS, SIZE

    RADIUS = 3
    SIZE = 512
    prompt = ('The task is to search for the best-performing combination of molecular descriptor and classification algorithm to build a predictive model for a chemical reaction. This reaction data is given by demo_test.csv and simplified demo_test.csv. '
              'Convert demo_test.csv to RDKit descriptors and Morgan fingerprints, and convert simplified demo_test.csv to simplified Morgan fingerprints. Then, using the four algorithms MLP, RandomForest, AdaBoost, and KNeighbors, combine the three Molecule descriptors to obtain the corresponding prediction results. '
              'Finally, give out which combination has the highest predicted value.')
    tools = ['ChemicalTools']

    run_langchain(prompt, tools)
    # print(load_and_prepare_Morganfingerprints_data('D:/PythonProject/Chem_AI/demo_test_morgan_fingerprints_3_512.csv'))
