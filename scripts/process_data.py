import pandas as pd 
import json
import os
from sklearn.preprocessing import LabelEncoder
import argparse

def read_csv(filepath, col_features, irrelevant_cols, irrelevant_rows, feature_index, group_index):
    '''
    col_features: True if cols are features, row instances
    irrelevant_cols: cols to delete, 1-based index
    irrelevant_rows: rows to delete, 1-based index
    feature_index: index of feature, 1-based index

    make sure there is no duplication in feature names
    '''
    
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata"))

    filename = os.path.splitext(os.path.basename(filepath))[0]

    df = pd.read_csv(filepath, header=None, low_memory=False)

    df.drop(index=[i-1 for i in irrelevant_rows], inplace=True)
    df.drop(columns=[i-1 for i in irrelevant_cols], inplace=True)

    # adjust feature index and group index 
    '''
    if col_features:
        feature_index -= len([i for i in irrelevant_rows if i < feature_index])
        group_index -= len([i for i in irrelevant_cols if i < group_index])
    else:
        feature_index -= len([i for i in irrelevant_cols if i < feature_index])
        group_index -= len([i for i in irrelevant_rows if i < group_index])
    '''
        
    # transpose
    if not col_features:
        df = df.transpose()

    labels = df.loc[:, group_index-1].to_list()
    labels = [str(l) for l in labels][1:]
    df.drop(columns=group_index-1, inplace=True)


    features = df.loc[feature_index-1, :].to_list()
    features = [str(f) for f in features][1:]
    df.drop(df.index[feature_index-1], inplace=True)

    # reset index 
    df.reset_index(drop=True, inplace=True)
    df.columns = range(df.shape[1])
    
    # Drop missing values

    # Reset row index 

    instances = []

    #print(features)
    #print("--------", labels)
    #print(df)

    for idx, row in df.iterrows():
        f_values = row.to_list()
        label = labels[idx]

        f_dict = {}
        for f_name, f_value in zip(features, f_values):
            try:
                f_dict[f_name] = float(f_value)
            except ValueError:
                raise ValueError(f"Cannot convert value '{f_value}' to float at row {idx+1}")

        instances.append({
            "Features": f_dict, 
            "Label": label
        })
    
    dataset = {
        'Instance': instances, 
        "Features": features,
        "Label": filename
    }
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    with open(f"{filename}.json", "w") as f:
        json.dump(dataset, f)

    with open(f"{filename}_labels.json", "w") as f:
        json.dump(labels.tolist(), f)
    

def str2bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def str2list(arg):
    if not arg.strip():
        print("HIIII")
        return []
    else:
        return list(map(int, map(str.strip, arg.split(","))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the CSV file")
    parser.add_argument("filepath", type=str, help="Path to the CSV file")
    parser.add_argument("col_features", type=str2bool, help="Are cols features")
    parser.add_argument("irrelevant_cols", type=str2list, help="Cols to delete, 1-based index")
    parser.add_argument("irrelevant_rows", type=str2list, help="Rows to delete, 1-based index")
    parser.add_argument("feature_index", type=str, help="Feature index")
    parser.add_argument("group_index", type=str, help="Group index")
    args = parser.parse_args()

    read_csv(
        args.filepath, 
        args.col_features, 
        args.irrelevant_cols, 
        args.irrelevant_rows, 
        int(args.feature_index), 
        int(args.group_index)
    )