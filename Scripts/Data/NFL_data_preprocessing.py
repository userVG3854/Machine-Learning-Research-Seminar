import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
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

def preprocess_nfl2(dataframes, verbose=False):
    preprocessed_dataframes = {}

    for df_name, tracking_data in dataframes.items():
        # Feature selection and preprocessing
        selected_features = ['gameId','frame', 'x', 'y', 's', 'o', 'dir', 'playDirection', 'displayName', 'teamAbbr', 'position']
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
        tracking_data = pd.get_dummies(tracking_data, columns=['playDirection'])

        # Use iterative imputer to handle missing values with emphasis on the temporal context
        numerical_features = ['x', 'y', 's', 'o', 'dir']
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

def preprocess_nfl(dataframes, verbose=False):
    preprocessed_dataframes = {}

    for df_name, tracking_data in dataframes.items():
        # Feature selection and preprocessing
        selected_features = ['gameId','frame', 'x', 'y', 's', 'o', 'dir', 'playDirection', 'displayName', 'teamAbbr', 'position']
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

        # Move playerId column to the desired location
        cols = list(tracking_data.columns)
        cols.insert(cols.index('frame'), cols.pop(cols.index('playerId')))
        tracking_data = tracking_data[cols]

        # Convert categorical features to numerical representations
        tracking_data = pd.get_dummies(tracking_data, columns=['playDirection'])

        # Use iterative imputer to handle missing values with emphasis on the temporal context
        numerical_features = ['x', 'y', 's', 'o', 'dir']
        imputer = IterativeImputer(max_iter=10, random_state=0)  # Adjust parameters as needed
        for group_name, group_df in tracking_data.groupby(['gameId', 'frame']):
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
    X_train, Y_train, Y_train_noised = {}, {}, {}
    X_test, Y_test, Y_test_noised = {}, {}, {}
    Y_noise_train, Y_noise_test, Data_train, Data_test = {}, {}, {}, {}
    X_noise_train, X_noise_test = {}, {}
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
        noise = noisy_df-df.values

        try:
            # Reshape the data to the format (number_frames, number_players, number_features)
            num_frames = int(df.shape[0] / 23)
            num_features = df.shape[1]
            x = df.values.reshape(num_frames, 23, num_features)
            y = y.reshape(num_frames, 23, num_features)
            noise = noise.reshape(num_frames, 23, num_features)

            # Remove one frame if the number of frames is odd
            if num_frames % 2 != 0:
                x = x[:-1]  # Remove the last frame
                y = y[:-1]  # Remove the last frame
                noise = noise[:-1]

            # Split the data into two equal parts
            split_index = int(num_frames * 0.5)

            x_train, y_target = x[:split_index], x[split_index:]
            y_noised, y_target_noised = y[:split_index], y[split_index:]
            noise_x, noise_y = noise[:split_index], noise[split_index:]

            # Append to respective dictionaries
            Data_train[df_name] = torch.tensor(x, dtype=torch.float32)
            X_train[df_name] = torch.tensor(x_train, dtype=torch.float32)
            Y_train[df_name] = torch.tensor(y_target, dtype=torch.float32)
            Y_train_noised[df_name] = torch.tensor(y_target_noised, dtype=torch.float32)
            Y_noise_train[df_name] = torch.tensor(noise_y, dtype=torch.float32)
            X_noise_train[df_name] = torch.tensor(noise_x, dtype=torch.float32)
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
        noise = noisy_df - test_df.values
        num_frames = int(test_df.shape[0] / 23)
        num_features = test_df.shape[1]
        x = test_df.values.reshape(num_frames, 23, num_features)
        y = y.reshape(num_frames, 23, num_features)
        noise = noise.reshape(num_frames, 23, num_features)

        # Remove one frame if the number of frames is odd
        if num_frames % 2 != 0:
            x = x[:-1]  # Remove the last frame
            y = y[:-1]  # Remove the last frame
            noise = noise[:-1]

        # Split the data into two equal parts
        split_index = int(num_frames * 0.5)

        x_test, y_target = x[:split_index], x[split_index:]
        y_noised, y_target_noised = y[:split_index], y[split_index:]
        noise_x, noise_y = noise[:split_index], noise[split_index:]

        # Append to respective dictionaries
        Data_test[test_game] = torch.tensor(x, dtype=torch.float32)
        X_test[test_game] = torch.tensor(x_test, dtype=torch.float32)
        Y_test[test_game] = torch.tensor(y_target, dtype=torch.float32)
        Y_test_noised[test_game] = torch.tensor(y_target_noised, dtype=torch.float32)
        Y_noise_test[test_game] = torch.tensor(noise_y, dtype=torch.float32)
        X_noise_test[test_game] = torch.tensor(noise_x, dtype=torch.float32)

    return X_train, Y_train, Y_train_noised, Y_noise_train, X_noise_train, X_test, Y_test, Y_test_noised, Y_noise_test, X_noise_test, Data_train, Data_test



def normalization2(X_train, Y_train, Y_train_noised, Y_noise_train, X_noise_train, X_test, Y_test, Y_test_noised, Y_noise_test, X_noise_test, Data_train, Data_test):
    # Initialize the dictionaries for the normalized data
    X_train_normalized = {}
    Y_train_normalized = {}
    Y_train_noised_normalized = {}
    Y_noise_train_normalized = {}
    X_noise_train_normalized = {}
    Data_train_normalized = {}
    X_test_normalized = {}
    Y_test_normalized = {}
    Y_test_noised_normalized = {}
    Y_noise_test_normalized = {}
    X_noise_test_normalized = {}
    Data_test_normalized = {}

    # Initialize the scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_y_noise = StandardScaler()
    scaler_x_noise = StandardScaler()

    # Normalize the training data
    for game, x in X_train.items():
        y = Y_train[game]
        y_noised = Y_train_noised[game]
        y_noise = Y_noise_train[game]
        x_noise = X_noise_train[game]
        data = Data_train[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Fit the scalers to the flattened tensors
        scaler_x.fit(x_flattened)
        scaler_y.fit(y_flattened)
        scaler_y_noise.fit(y_noise_flattened)
        scaler_x_noise.fit(x_noise_flattened)

        # Transform the flattened tensors
        x_normalized_np = scaler_x.transform(x_flattened)
        y_normalized_np = scaler_y.transform(y_flattened)
        y_noise_normalized_np = scaler_y_noise.transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.transform(x_noise_flattened)
        data_normalized_np = scaler_x.transform(data_flattened)

        # Derive y_noised_normalized from y_normalized and y_noise_normalized
        y_noised_normalized_np = y_normalized_np + y_noise_normalized_np

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_train_normalized[game] = x_normalized
        Y_train_normalized[game] = y_normalized
        Y_train_noised_normalized[game] = y_noised_normalized
        Y_noise_train_normalized[game] = y_noise_normalized
        X_noise_train_normalized[game] = x_noise_normalized
        Data_train_normalized[game] = data_normalized

    # Normalize the test data
    for game, x in X_test.items():
        y = Y_test[game]
        y_noised = Y_test_noised[game]
        y_noise = Y_noise_test[game]
        x_noise = X_noise_test[game]
        data = Data_test[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Transform the flattened tensors
        x_normalized_np = scaler_x.transform(x_flattened)
        y_normalized_np = scaler_y.transform(y_flattened)
        y_noise_normalized_np = scaler_y_noise.transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.transform(x_noise_flattened)
        data_normalized_np = scaler_x.transform(data_flattened)

        # Derive y_noised_normalized from y_normalized and y_noise_normalized
        y_noised_normalized_np = y_normalized_np + y_noise_normalized_np

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_test_normalized[game] = x_normalized
        Y_test_normalized[game] = y_normalized
        Y_test_noised_normalized[game] = y_noised_normalized
        Y_noise_test_normalized[game] = y_noise_normalized
        X_noise_test_normalized[game] = x_noise_normalized
        Data_test_normalized[game] = data_normalized

    return X_train_normalized, Y_train_normalized, Y_train_noised_normalized, Y_noise_train_normalized, X_noise_train_normalized, X_test_normalized, Y_test_normalized, Y_test_noised_normalized, Y_noise_test_normalized, X_noise_test_normalized, Data_train_normalized, Data_test_normalized, scaler_x, scaler_y, scaler_y_noise, scaler_x_noise


def normalization(X_train, Y_train, Y_train_noised, Y_noise_train, X_noise_train, X_test, Y_test, Y_test_noised, Y_noise_test, X_noise_test, Data_train, Data_test): 
    # Initialize the dictionaries for the normalized data
    X_train_normalized = {}
    Y_train_normalized = {}
    Y_train_noised_normalized = {}
    Y_noise_train_normalized = {}
    X_noise_train_normalized = {}
    Data_train_normalized = {}
    X_test_normalized = {}
    Y_test_normalized = {}
    Y_test_noised_normalized = {}
    Y_noise_test_normalized = {}
    X_noise_test_normalized = {}
    Data_test_normalized = {}

    # Initialize the scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_y_noised = StandardScaler()
    scaler_y_noise = StandardScaler()
    scaler_x_noise = StandardScaler()
    scaler_data = StandardScaler()

    # Normalize the training data
    for game, x in X_train.items():
        y = Y_train[game]
        y_noised = Y_train_noised[game]
        y_noise = Y_noise_train[game]
        x_noise = X_noise_train[game]
        data = Data_train[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Fit the scalers to the flattened tensors and transform
        x_normalized_np = scaler_x.fit_transform(x_flattened)
        y_normalized_np = scaler_y.fit_transform(y_flattened)
        y_noised_normalized_np = scaler_y_noised.fit_transform(y_noised_flattened)
        y_noise_normalized_np = scaler_y_noise.fit_transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.fit_transform(x_noise_flattened)
        data_normalized_np = scaler_data.fit_transform(data_flattened)

        #y_noised_normalized_np = y_normalized_np + y_noise_normalized_np

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_train_normalized[game] = x_normalized
        Y_train_normalized[game] = y_normalized
        Y_train_noised_normalized[game] = y_noised_normalized
        Y_noise_train_normalized[game] = y_noise_normalized
        X_noise_train_normalized[game] = x_noise_normalized
        Data_train_normalized[game] = data_normalized

    # Normalize the test data
    for game, x in X_test.items():
        y = Y_test[game]
        y_noised = Y_test_noised[game]
        y_noise = Y_noise_test[game]
        x_noise = X_noise_test[game]
        data = Data_test[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Fit the scalers to the flattened tensors and transform
        x_normalized_np = scaler_x.fit_transform(x_flattened)
        y_normalized_np = scaler_y.fit_transform(y_flattened)
        y_noised_normalized_np = scaler_y_noised.fit_transform(y_noised_flattened)
        y_noise_normalized_np = scaler_y_noise.fit_transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.fit_transform(x_noise_flattened)
        data_normalized_np = scaler_data.fit_transform(data_flattened)

        #y_noised_normalized_np = y_normalized_np + y_noise_normalized_np

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_test_normalized[game] = x_normalized
        Y_test_normalized[game] = y_normalized
        Y_test_noised_normalized[game] = y_noised_normalized
        Y_noise_test_normalized[game] = y_noise_normalized
        X_noise_test_normalized[game] = x_noise_normalized
        Data_test_normalized[game] = data_normalized
    

    return X_train_normalized, Y_train_normalized, Y_train_noised_normalized, Y_noise_train_normalized, X_noise_train_normalized, X_test_normalized, Y_test_normalized, Y_test_noised_normalized, Y_noise_test_normalized, X_noise_test_normalized, Data_train_normalized, Data_test_normalized, scaler_x, scaler_y, scaler_y_noised, scaler_y_noise, scaler_x_noise, scaler_data







def normalization2(X_train, Y_train, Y_train_noised, Y_noise_train, X_noise_train, X_test, Y_test, Y_test_noised, Y_noise_test, X_noise_test, Data_train, Data_test): 
    # Initialize the dictionaries for the normalized data
    X_train_normalized = {}
    Y_train_normalized = {}
    Y_train_noised_normalized = {}
    Y_noise_train_normalized = {}
    X_noise_train_normalized = {}
    Data_train_normalized = {}
    X_test_normalized = {}
    Y_test_normalized = {}
    Y_test_noised_normalized = {}
    Y_noise_test_normalized = {}
    X_noise_test_normalized = {}
    Data_test_normalized = {}

    # Initialize the scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_y_noised = StandardScaler()
    scaler_y_noise = StandardScaler()
    scaler_x_noise = StandardScaler()
    scaler_data = StandardScaler()

    # Normalize the training data
    for game, x in X_train.items():
        y = Y_train[game]
        y_noised = Y_train_noised[game]
        y_noise = Y_noise_train[game]
        x_noise = X_noise_train[game]
        data = Data_train[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Fit the scalers to the flattened tensors and transform
        x_normalized_np = scaler_x.fit_transform(x_flattened)
        y_normalized_np = scaler_y.fit_transform(y_flattened)
        y_noised_normalized_np = scaler_y_noised.fit_transform(y_noised_flattened)
        y_noise_normalized_np = scaler_y_noise.fit_transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.fit_transform(x_noise_flattened)
        data_normalized_np = scaler_data.fit_transform(data_flattened)

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_train_normalized[game] = x_normalized
        Y_train_normalized[game] = y_normalized
        Y_train_noised_normalized[game] = y_noised_normalized
        Y_noise_train_normalized[game] = y_noise_normalized
        X_noise_train_normalized[game] = x_noise_normalized
        Data_train_normalized[game] = data_normalized

    # Normalize the test data
    for game, x in X_test.items():
        y = Y_test[game]
        y_noised = Y_test_noised[game]
        y_noise = Y_noise_test[game]
        x_noise = X_noise_test[game]
        data = Data_test[game]

        # Flatten the tensors along the frame and player dimensions
        x_flattened = x.view(x.shape[0] * x.shape[1], -1).numpy()
        y_flattened = y.view(y.shape[0] * y.shape[1], -1).numpy()
        y_noised_flattened = y_noised.view(y_noised.shape[0] * y_noised.shape[1], -1).numpy()
        y_noise_flattened = y_noise.view(y_noise.shape[0] * y_noise.shape[1], -1).numpy()
        x_noise_flattened = x_noise.view(x_noise.shape[0] * x_noise.shape[1], -1).numpy()
        data_flattened = data.view(data.shape[0]*data.shape[1], -1).numpy()

        # Fit the scalers to the flattened tensors and transform
        x_normalized_np = scaler_x.fit_transform(x_flattened)
        y_normalized_np = scaler_y.fit_transform(y_flattened)
        y_noised_normalized_np = scaler_y_noised.fit_transform(y_noised_flattened)
        y_noise_normalized_np = scaler_y_noise.fit_transform(y_noise_flattened)
        x_noise_normalized_np = scaler_x_noise.fit_transform(x_noise_flattened)
        data_normalized_np = scaler_data.fit_transform(data_flattened)

        # Convert the normalized arrays back to tensors and reshape to original shape
        x_normalized = torch.tensor(x_normalized_np, dtype=torch.float32).view(x.shape)
        y_normalized = torch.tensor(y_normalized_np, dtype=torch.float32).view(y.shape)
        y_noised_normalized = torch.tensor(y_noised_normalized_np, dtype=torch.float32).view(y_noised.shape)
        y_noise_normalized = torch.tensor(y_noise_normalized_np, dtype=torch.float32).view(y_noise.shape)
        x_noise_normalized = torch.tensor(x_noise_normalized_np, dtype=torch.float32).view(x_noise.shape)
        data_normalized = torch.tensor(data_normalized_np, dtype=torch.float32).view(data.shape)

        # Append to the dictionaries
        X_test_normalized[game] = x_normalized
        Y_test_normalized[game] = y_normalized
        Y_test_noised_normalized[game] = y_noised_normalized
        Y_noise_test_normalized[game] = y_noise_normalized
        X_noise_test_normalized[game] = x_noise_normalized
        Data_test_normalized[game] = data_normalized
    

    return X_train_normalized, Y_train_normalized, Y_train_noised_normalized, Y_noise_train_normalized, X_noise_train_normalized, X_test_normalized, Y_test_normalized, Y_test_noised_normalized, Y_noise_test_normalized, X_noise_test_normalized, Data_train_normalized, Data_test_normalized, scaler_x, scaler_y, scaler_y_noised, scaler_y_noise, scaler_x_noise, scaler_data













