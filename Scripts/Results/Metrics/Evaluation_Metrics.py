import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

def min_ade20(pred_trajs, gt_trajs):
    """Compute minimum average displacement error over the top 20 predicted trajectories."""
    ade_list = []
    for pred_traj in pred_trajs:
        ade = np.sqrt(np.mean(np.square(pred_traj - gt_trajs)))
        ade_list.append(ade)
    ade_list.sort()
    return np.mean(ade_list[:20])

def min_fde20(pred_trajs, gt_trajs):
    """Compute minimum final displacement error over the top 20 predicted trajectories."""
    fde_list = []
    for pred_traj in pred_trajs:
        fde = np.sqrt(np.mean(np.square(pred_traj[-1] - gt_trajs[-1])))
        fde_list.append(fde)
    fde_list.sort()
    return np.mean(fde_list[:20])


def reverse_diffusion_process(model, y_test, noise_predictions):
    y_pred = y_test.clone()
    for t in range(y_test.shape[0]):
        with torch.no_grad():
            # Get the noise prediction for the current timestep
            noise_pred = noise_predictions[t, :, :]

            # Compute the mean and variance for the current timestep
            sqrt_alpha = model.ddpm.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = model.ddpm.sqrt_one_minus_alphas_cumprod[t]
            posterior_mean = (sqrt_alpha * (y_test - sqrt_one_minus_alpha * noise_pred)) / (sqrt_alpha**2 + sqrt_one_minus_alpha**2)
            posterior_variance = model.ddpm.q_posterior_variance(torch.tensor([t]).to(y_test.device))

            # Sample from the posterior distribution
            eps = torch.randn_like(y_pred)
            y_pred = posterior_mean + torch.sqrt(posterior_variance) * eps

    return y_pred










def train_lstm(model, X_train_normalized, Y_train_normalized, epochs, learning_rate, batch_size, X_test_normalized, Y_test_normalized, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    model.train()
    best_val_loss = float('inf')
    no_improvement = 0
    patience = 3

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        for df_name, x in X_train_normalized.items():
            y = Y_train_normalized[df_name]
            for i in range(0, len(x), batch_size):
                optimizer.zero_grad()
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                output = model(batch_x)

                # Use the actual trajectory as the target
                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(X_train_normalized)

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_df_name, val_x in X_test_normalized.items():
                val_y = Y_test_normalized[val_df_name]
                for i in range(0, len(val_x), batch_size):
                    val_batch_x = val_x[i:i+batch_size]
                    val_batch_y = val_y[i:i+batch_size]
                    val_output = model(val_batch_x)

                    # Use the actual trajectory as the target
                    val_loss += criterion(val_output, val_batch_y).item()

            val_loss /= len(X_test_normalized)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f},  Time: {time.time() - start_time:.4f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), model_name)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(val_loss)

    model.eval()



def train_diff(model, X_train, Y_train, epochs, learning_rate, batch_size, X_test, Y_test, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    model.train()
    best_val_loss = float('inf')
    no_improvement = 0
    patience = 3

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        for df_name, x in X_train.items():
            y = Y_train[df_name]
            for i in range(0, len(x), batch_size):
                optimizer.zero_grad()
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                output = model(batch_x, batch_y)

                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(X_train)

        # Evaluation on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_game, x in X_test.items():
                x = X_test[test_game]
                y = Y_test[test_game]
                for i in range(0, len(x), batch_size):
                    batch_x = x[i:i+batch_size]
                    batch_y = y[i:i+batch_size]
                    output = model(batch_x, batch_y)
                    loss = criterion(output, batch_y)

                    test_loss += loss.item()

            test_loss /= len(X_test)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f},  Time: {time.time() - start_time:.4f}s")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            no_improvement = 0
            torch.save(model.state_dict(), model_name)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        scheduler.step(test_loss)

    model.eval()