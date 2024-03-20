import pandas as pd
import torch
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import numpy as np

def open_all_tsv(folder_path):
    dataframes = {}
    for file in os.listdir(folder_path):
        if file.endswith(".tsv"):
            df_name = os.path.splitext(file)[0]  # Get the file name without extension
            dataframes[df_name] = pd.read_csv(os.path.join(folder_path, file), sep="\t")
    print("Conversion completed for all TSV files in the folder.")        
    return dataframes





def get_team_type(team_data):
    position_counts = team_data['position'].value_counts()
    if 'QB' in position_counts and position_counts['QB'] > 0:
        return 'attacking'
    elif 'ball' in position_counts:
        return 'ball'
    else:
        return 'defensive'

def preprocess_nfl(dataframes, verbose=False):
    preprocessed_dataframes = {}

    for df_name, tracking_data in dataframes.items():
        # Feature selection and preprocessing
        selected_features = ['gameId', 'playId', 'playType', 'frame', 'x', 'y', 's', 'o', 'dir', 'playDirection', 'quarter', 'down', 'yardsToGo', 'displayName', 'teamAbbr', 'position']
        tracking_data = tracking_data[selected_features]

        # Drop rows with excess frames based on playerId
        frame_counts = tracking_data.groupby('displayName')['frame'].count()
        if not frame_counts.unique().size == 1:
            min_frames = frame_counts.min()
            tracking_data = tracking_data.groupby('displayName').apply(lambda x: x.head(min_frames))



        # Fill missing values for specific columns based on displayName
        mask = tracking_data['displayName'] == 'ball'
        tracking_data.loc[mask, 'teamAbbr'] = 'BALL'
        tracking_data.loc[mask, 'position'] = 'ball'
        tracking_data.loc[mask, 'o'] = 0
        tracking_data.loc[mask, 'dir'] = 0

        # Determine team types and sort
        teams = tracking_data.groupby('teamAbbr')
        team_types = {team: get_team_type(team_data) for team, team_data in teams}
        tracking_data['teamType'] = tracking_data['teamAbbr'].map(team_types)

        # Fill NaN values in 'teamType' column
        tracking_data['teamType'].fillna('ball', inplace=True)
        team_type_counts = tracking_data['teamType'].value_counts()
        if verbose:
            print(f"Count of each team type:\n{team_type_counts}")

        # Create playerId based on teamType and position
        tracking_data = tracking_data.sort_values(['teamType', 'position'])
        tracking_data['playerId'] = 11 * (tracking_data['teamType'] == 'attacking').astype(int) + 12 * (tracking_data['teamType'] == 'defensive').astype(int) + 23 * (tracking_data['teamType'] == 'ball').astype(int) + tracking_data.groupby(['teamType', 'position']).cumcount() + 1

        # Convert categorical features to numerical representations
        tracking_data = pd.get_dummies(tracking_data, columns=['playType', 'playDirection', 'teamType'])

        # Use iterative imputer to handle missing values with emphasis on the temporal context
        numerical_features = ['x', 'y', 's', 'o', 'dir', 'quarter', 'down', 'yardsToGo']
        imputer = IterativeImputer(max_iter=10, random_state=0)  # Adjust parameters as needed
        for group_name, group_df in tracking_data.groupby(['gameId', 'playId']):
            imputed = imputer.fit_transform(group_df[numerical_features])
            tracking_data.loc[group_df.index, numerical_features] = pd.DataFrame(imputed, columns=numerical_features, index=group_df.index)

        columns_with_nan = tracking_data.columns[tracking_data.isna().any()]
        if columns_with_nan.any():
            if verbose:
                print(f"Columns with NaN values: {columns_with_nan}")
                print(f"Dropped game with remaining nan values: {df_name}")
            continue
        else:
            # Keep only numerical columns
            numerical_columns = tracking_data.select_dtypes(include=[np.number]).columns
            tracking_data = tracking_data[numerical_columns]
            preprocessed_dataframes[df_name] = tracking_data

    return preprocessed_dataframes




def check_dataframe_shapes(preprocessed_dataframes):
    num_expected_shapes = 0
    num_unexpected_shapes = 0

    for df_name in list(preprocessed_dataframes.keys()):  # Create a copy of the keys
        df = preprocessed_dataframes[df_name]
        num_frames = df['frame'].max() + 1
        num_actors = 23  # 11 players for each team and 1 for the ball
        expected_rows = num_frames * num_actors
        expected_cols = df.shape[1]  # Assuming all dataframes have the same number of columns

        if df.shape != (expected_rows, expected_cols):
            print(f"Dataframe '{df_name}' has an unexpected shape: {df.shape} (expected: {(expected_rows, expected_cols)})")
            del preprocessed_dataframes[df_name]
            num_unexpected_shapes += 1
        else:
            print(f"Dataframe '{df_name}' has the expected shape: {df.shape}")
            num_expected_shapes += 1

    print(f"Total number of dataframes with expected shape: {num_expected_shapes}")
    print(f"Total number of dataframes with unexpected shape: {num_unexpected_shapes}")
    print(f"Total number of dataframes: {len(preprocessed_dataframes)}")

    return preprocessed_dataframes




def prepare_data(dataframes, test_games):
    X_train, Y_train = {}, {}
    X_test, Y_test = {}, {}
    num_success = 0
    num_fail = 0

    # Noise process parameters
    N = 1000
    gamma_min = 0.1
    gamma_max = 0.9
    gammas = np.linspace(gamma_max, gamma_min, N)

    for df_name, df in dataframes.items():
        # Skip the test game in the training set creation
        if df_name in test_games:
            continue

        # Reset the index of the DataFrame if 'playerId' is an index
        if 'playerId' in df.index.names:
            df.reset_index('playerId', drop=True, inplace=True)

        # Keep only numeric columns in the dataframe
        df = df.select_dtypes(include=[np.number])

        # Apply the noise process to the dataframe
        noisy_df = np.copy(df.values)
        for t in range(1, len(df)):
            k = min(t, N-1)  # Ensure not to exceed the number of defined gammas
            gamma = gammas[k]
            Z_t = np.random.normal(0, 1, size=df.iloc[t].shape)
            noisy_df[t] = gamma * df.iloc[t - 1] + np.sqrt(1 - gamma**2) * Z_t

        # Calculate the difference between the original and noisy trajectories
        y = noisy_df

        try:
            # Reshape the data to the format (number_frames, number_players, number_features)
            num_frames = int(df.shape[0] / 23)
            num_features = df.shape[1]
            x = df.values.reshape(num_frames, 23, num_features)
            y = y.reshape(num_frames, 23, num_features)

            # Append to respective dictionaries
            X_train[df_name] = torch.tensor(x, dtype=torch.float32)
            Y_train[df_name] = torch.tensor(y, dtype=torch.float32)
            num_success += 1
        except ValueError:
            print(f"Failed to reshape dataframe '{df_name}'")
            num_fail += 1

    print(f"Number of successfully converted dataframes: {num_success}")
    print(f"Number of failed conversions: {num_fail}")

    # Prepare the test set
    for test_game in test_games:
        test_df = dataframes[test_game]
    
        if 'playerId' in test_df.index.names:
            test_df.reset_index('playerId', drop=True, inplace=True)
        test_df = test_df.select_dtypes(include=[np.number])
        noisy_df = np.copy(test_df.values)
        for t in range(1, len(test_df)):
            k = min(t, N-1)
            gamma = gammas[k]
            Z_t = np.random.normal(0, 1, size=test_df.iloc[t].shape)
            noisy_df[t] = gamma * test_df.iloc[t - 1] + np.sqrt(1 - gamma**2) * Z_t

        y = noisy_df
        num_frames = int(test_df.shape[0] / 23)
        num_features = test_df.shape[1]
        x = test_df.values.reshape(num_frames, 23, num_features)
        y = y.reshape(num_frames, 23, num_features)

        # Append to respective dictionaries
        X_test[test_game] = torch.tensor(x, dtype=torch.float32)
        Y_test[test_game] = torch.tensor(y, dtype=torch.float32)

    return X_train, Y_train, X_test, Y_test



def prepare_data2(dataframes, test_games):
    X_train, Y_train, X_test, Y_test = [], [], [], []
    num_success = 0
    num_fail = 0

    # Noise process parameters
    N = 1000
    gamma_min = 0.1
    gamma_max = 0.9
    gammas = np.linspace(gamma_max, gamma_min, N)

    for df_name, df in dataframes.items():
        # Skip the test game in the training set creation
        if df_name == test_game:
            continue

        # Reset the index of the DataFrame if 'playerId' is an index
        if 'playerId' in df.index.names:
            df.reset_index('playerId', drop=True, inplace=True)

        # Keep only numeric columns in the dataframe
        df = df.select_dtypes(include=[np.number])

        # Apply the noise process to the dataframe
        noisy_df = np.copy(df.values)
        for t in range(1, len(df)):
            k = min(t, N-1)  # Ensure not to exceed the number of defined gammas
            gamma = gammas[k]
            Z_t = np.random.normal(0, 1, size=df.iloc[t].shape)
            noisy_df[t] = gamma * df.iloc[t - 1] + np.sqrt(1 - gamma**2) * Z_t

        # Calculate the difference between the original and noisy trajectories
        y = noisy_df

        try:
            # Reshape the data to the format (number_frames, number_players, number_features)
            num_frames = int(df.shape[0] / 23)
            num_features = df.shape[1]
            x = df.values.reshape(num_frames, 23, num_features)
            y = y.reshape(num_frames, 23, num_features)

            # Append to respective lists
            X_train.append(x)
            Y_train.append(y)
            num_success += 1
        except ValueError:
            print(f"Failed to reshape dataframe '{df_name}'")
            num_fail += 1

    print(f"Number of successfully converted dataframes: {num_success}")
    print(f"Number of failed conversions: {num_fail}")

    # Prepare the test set
    for test_game in test_games:
        test_df = dataframes[test_game]
    
        if 'playerId' in test_df.index.names:
            test_df.reset_index('playerId', drop=True, inplace=True)
        test_df = test_df.select_dtypes(include=[np.number])
        noisy_df = np.copy(test_df.values)
        for t in range(1, len(test_df)):
            k = min(t, N-1)
            gamma = gammas[k]
            Z_t = np.random.normal(0, 1, size=test_df.iloc[t].shape)
            noisy_df[t] = gamma * test_df.iloc[t - 1] + np.sqrt(1 - gamma**2) * Z_t

        y = noisy_df
        num_frames = int(test_df.shape[0] / 23)
        num_features = test_df.shape[1]
        x = test_df.values.reshape(num_frames, 23, num_features)
        y = y.reshape(num_frames, 23, num_features)

        X_test.append(x)
        Y_test.append(y)

    # Convert the lists of arrays to tensors
    X_train = [torch.tensor(x, dtype=torch.float32) for x in X_train]
    Y_train = [torch.tensor(y, dtype=torch.float32) for y in Y_train]
    X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
    Y_test = [torch.tensor(y, dtype=torch.float32) for y in Y_test]

    return X_train, Y_train, X_test, Y_test














