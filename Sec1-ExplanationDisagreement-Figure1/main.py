import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency
import os
from sklearn.neighbors import NearestNeighbors

from lime import lime_tabular

def download_pima_indians_dataset():
    """
    Downloads the Pima Indians Diabetes dataset and returns the features and labels.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 'Diabetes\nPedigree\nFunction', 'Age', 'Outcome']
    diabetes_df = pd.read_csv(url, names=column_names)
    
    x_data = diabetes_df.iloc[:, :-1].values
    y_data = diabetes_df.iloc[:, -1].values
    
    return x_data, y_data

def train_neural_network(x_train, y_train, input_size, hidden_size, output_size, num_epochs=5000, learning_rate=0.01):
    """
    Trains a simple neural network classifier using PyTorch.
    """
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    class SimpleNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, output_size)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out

    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = torch.tensor(x_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def get_outputs(model, inputs, threshold=0.5):
    with torch.no_grad():
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        outputs = model(inputs_tensor)
        predicted = (outputs > threshold).float()
    return predicted

def evaluate_model(model, x_train, y_train, x_test, y_test):
    with torch.no_grad():
        # Evaluate on training data
        train_predicted = get_outputs(model, x_train)
        train_labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        train_accuracy = (train_predicted.eq(train_labels).sum().item()) / len(y_train)
        print(f'Train Accuracy: {train_accuracy:.4f}')

        # Evaluate on test data
        test_predicted = get_outputs(model, x_test)
        test_labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        test_accuracy = (test_predicted.eq(test_labels).sum().item()) / len(y_test)
        print(f'Test Accuracy: {test_accuracy:.4f}')

def plot_force_plot(base_value, shap_values, features, feature_names, index, caller):
    shap_values_sum = np.abs(shap_values).sum()
    normalized_shap_values = shap_values.copy()
    if shap_values_sum != 0:
        normalized_shap_values = shap_values * (100 / shap_values_sum)

    # Reduce the overall height of the plot
    shap.force_plot(
        base_value=base_value,
        shap_values=normalized_shap_values,
        features=features,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        out_names='',
        figsize=(10, 2.5),  # Reduced height from 4 to 3
        plot_cmap='RdBu',
        # text_rotation=45,
        contribution_threshold=0.2,
    )
    final_pred = float(np.round(normalized_shap_values.flatten().sum(),2))
    final_pred = round(final_pred, 2) # required for rounding GRAD !?
    
    for item in plt.gca().get_children():
        if isinstance(item, plt.Text) and str(final_pred) in item.get_text():
            item.remove()
        if isinstance(item, plt.Text) and "base value" in item.get_text():
            item.set_text(" Signed Feature Importance Percentages")
            item.set_fontsize(item.get_fontsize() + 6)  # Increase font size
            item.set_color('black')  # Set text color to black
            item.set_zorder(-1)  # Bring the text to the top
        if isinstance(item, plt.Text) and "higher" in item.get_text():
            item.set_text("Diabetic ")
            item.set_fontsize(item.get_fontsize() + 4)  # Increase font size
            item.set_zorder(10)  # Bring the text to the top
            # absc, ordi = item.get_position()
            # item.set_position((absc, ordi + 0.9))  # Move the text a little higher up
        if isinstance(item, plt.Text) and "lower" in item.get_text():
            item.set_text(" Non-diabetic")
            item.set_fontsize(item.get_fontsize() + 4)  # Increase font size
            item.set_zorder(10)  # Bring the text to the top
            # absc, ordi = item.get_position()
            # item.set_position((absc, ordi + 0.2))
        if isinstance(item, plt.Text) and '=' in item.get_text():
            item.set_fontsize(item.get_fontsize() + 3)
            # item.set_weight('bold')
            absc, ordi = item.get_position()
            if caller == 'grad' and 'BMI' in item.get_text():
                item.set_position((absc + 7.5, ordi - 0.2))
                pass
            elif caller == 'shap' and 'Glucose' in item.get_text():
                item.set_position((absc - 6.6, ordi - 0.2))
                pass
            elif caller == 'shap' and 'BMI' in item.get_text():
                item.set_position((absc - 5.5, ordi - 0.2))
                pass
            else:
                item.set_position((absc, ordi - 0.2))
    # plt.subplots_adjust(top=0.7, bottom=0.1)  # Adjust margins to reduce top and bottom space
    plt.tight_layout()
    plt.savefig(f'PLOTS/{caller}_force_plot_{index}.pdf', dpi=300)

def compute_ig_explanations(model, x_test, index, target_class_index=0):
    input_tensor = torch.tensor(x_test[index], dtype=torch.float32).unsqueeze(0)
    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute(input_tensor, target=target_class_index, return_convergence_delta=True)
    return attributions.detach().numpy().reshape(1, -1)

def compute_grad_explanations(model, x_test, index, target_class_index=0):
    input_tensor = torch.tensor(x_test[index], dtype=torch.float32).unsqueeze(0)
    grad = Saliency(model)
    attributions = grad.attribute(input_tensor, target=target_class_index, abs=False)
    return attributions.detach().numpy().reshape(1, -1)

def compute_shap_explanations(model, x_train, x_test, index):
    model_fn = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    explainer = shap.KernelExplainer(model_fn, x_train)
    shap_values = explainer.shap_values(x_test)[index]
    return shap_values.reshape(1, -1)

def compute_lime_explanations(model, x_train, x_test, feature_names, index):
    model_fn = lambda x: np.hstack((1 - model(torch.tensor(x, dtype=torch.float32)).detach().numpy(), 
                                    model(torch.tensor(x, dtype=torch.float32)).detach().numpy()))
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=x_train,
        mode='classification',
        feature_names=feature_names,
        class_names=['healthy', 'diabetic'],
        discretize_continuous=False,
        sample_around_instance=True
    )
    exp = explainer.explain_instance(x_test[index], model_fn, labels=[1])
    lime_values = exp.as_list(label=1)
    lime_values_dict = dict(lime_values)

    lime_values_array = []
    for name in feature_names:
        value = 0
        for key, val in lime_values_dict.items():
            if key.startswith(name):
                value = val
                break
        lime_values_array.append(value)
    lime_values_array = np.array(lime_values_array)
    return lime_values_array

def save_ig_force_plot(model, x_test, feature_names, index=0):
    attributions = compute_ig_explanations(model, x_test, index)
    # Plot the force plot for the first datapoint using IG values
    plot_force_plot(
        base_value=0,
        shap_values=attributions,
        features=x_test[index],
        feature_names=feature_names,
        index=index,
        caller='ig'
    )


def save_grad_force_plot(model, x_test, feature_names, index=0):
    attributions = compute_grad_explanations(model, x_test, index)
    # Plot the force plot for the first datapoint using gradient values
    plot_force_plot(
        base_value=0,
        shap_values=attributions,
        features=x_test[index],
        feature_names=feature_names,
        index=index,
        caller='grad'
    )
    
def save_shap_force_plot(model, x_train, x_test, feature_names, index=0):
    shap_values = compute_shap_explanations(model, x_train, x_test, index)
    # Plot the force plot for the first datapoint
    plot_force_plot(
        base_value=0,
        shap_values=shap_values,
        features=x_test[index],
        feature_names=feature_names,
        index=index,
        caller='shap'
    )

def save_lime_force_plot(model, x_train, x_test, feature_names, index=0):
    lime_values_array = compute_lime_explanations(model, x_train, x_test, feature_names, index)

    # Plot the force plot for the first datapoint using LIME values
    plot_force_plot(
        base_value=0,
        shap_values=lime_values_array,
        features=x_test[index],
        feature_names=feature_names,
        index=index,
        caller='lime'
    )



def compute_exps(model, x_train, x_test, feature_names):
    # Check if the explanation files already exist
    grad_file = 'EXPLANATIONS/grad_explanations.csv'
    shap_file = 'EXPLANATIONS/shap_explanations.csv'
    lime_file = 'EXPLANATIONS/lime_explanations.csv'
    ig_file = 'EXPLANATIONS/ig_explanations.csv'
    
    if not (os.path.exists(grad_file) and os.path.exists(shap_file) and os.path.exists(lime_file) and os.path.exists(ig_file)):
        grad_df = pd.DataFrame(columns=feature_names)
        lime_df = pd.DataFrame(columns=feature_names)
        ig_df = pd.DataFrame(columns=feature_names)
        shap_df = pd.DataFrame(columns=feature_names)

        for index in range(len(x_test)):
            ig_explanations = compute_ig_explanations(model, x_test, index).flatten()
            grad_explanations = compute_grad_explanations(model, x_test, index).flatten()
            lime_explanations = np.array(compute_lime_explanations(model, x_train, x_test, feature_names, index)).flatten()
            shap_explanations = compute_shap_explanations(model, x_train, x_test, index).flatten()
            # shap_explanations = np.array([0,0,0,0,0,0,0,0]).flatten() # compute_shap_explanations(model, x_train, x_test, index).flatten()

            grad_df.loc[index] = grad_explanations
            shap_df.loc[index] = shap_explanations
            lime_df.loc[index] = lime_explanations
            ig_df.loc[index] = ig_explanations
        
        grad_df.to_csv(grad_file, index=False)
        shap_df.to_csv(shap_file, index=False)
        lime_df.to_csv(lime_file, index=False)
        ig_df.to_csv(ig_file, index=False)

    else:
        grad_df = pd.read_csv(grad_file)
        shap_df = pd.read_csv(shap_file)
        lime_df = pd.read_csv(lime_file)
        ig_df = pd.read_csv(ig_file)

    return grad_df, shap_df, lime_df, ig_df

def get_axe_pred(index, exp_df, data_df, preds, n=4, k=5):
    # Extract the index row from exp_df
    exp_row = exp_df.iloc[index]

    # Find the indices of the n largest absolute values from the row
    largest_indices = exp_row.abs().nlargest(n).index

    # Filter the data_df to keep only the columns with the largest indices
    filtered_data_df = data_df.loc[:, largest_indices]

    # Find the k nearest neighbors for the row at the given index
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(filtered_data_df)
    distances, indices = nbrs.kneighbors(filtered_data_df.iloc[[index]])

    # Get the k nearest neighbors and their corresponding predictions
    nearest_neighbors = filtered_data_df.iloc[indices[0]]
    nearest_neighbors_preds = preds[indices[0]]

    # # If index is 141, print the neighbors, distances, and corresponding predictions
    # if index == 141:
    #     print("Nearest Neighbors for index 141:")
    #     print(nearest_neighbors)
    #     print("Distances for index 141:")
    #     print(distances[0])
    #     print("Corresponding Predictions for index 141:")
    #     print(nearest_neighbors_preds)
    #     print("Corresponding Rows from data_df for index 141:")
    #     pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
    #     print(data_df.iloc[indices[0]])
    #     pd.reset_option('display.max_columns')  # Reset to default after printing

    return np.mean(nearest_neighbors_preds)
def get_all_axe(exp_dfs, data_df, preds, n=4, k=5):
    axe_results = []
    for exp_df in exp_dfs:
        axe_result = []
        for index in range(len(data_df)):
            axe_pred = get_axe_pred(index, exp_df, data_df, preds, n, k)
            axe_result.append(float(axe_pred))
        axe_results.append(axe_result)

    return axe_results

def plot_all(model, x_train, x_test, feature_names, index):
    save_grad_force_plot(model, x_test, feature_names, index=index)
    save_ig_force_plot(model, x_test, feature_names, index=index)
    save_lime_force_plot(model, x_train, x_test, feature_names, index=index)
    save_shap_force_plot(model, x_train, x_test, feature_names, index=index)
    # creates 4 files: 'PLOTS/grad_force_plot_{index}.png', 'PLOTS/ig_force_plot_{index}.png', 'PLOTS/lime_force_plot_{index}.png', and 'PLOTS/shap_force_plot_{index}.png'

def main():
    x_data, y_data = download_pima_indians_dataset()
    feature_names = ['Pregnancies', 'Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 'Diabetes Pedigree\nFunction', 'Age']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    input_size = x_train.shape[1]
    hidden_size = 10
    output_size = 1

    model = train_neural_network(x_train, y_train, input_size, hidden_size, output_size)

    # Evaluate the model
    evaluate_model(model, x_train, y_train, x_test, y_test)
    
    y_pred_tensor = get_outputs(model, x_test)
    y_pred = pd.Series(y_pred_tensor.numpy().flatten())

    grad_exps, shap_exps, lime_exps, ig_exps = compute_exps(model, x_train, x_test, feature_names)
    x_test_df = pd.DataFrame(x_test, columns=feature_names)
    grad_axe, shap_axe, lime_axe, ig_axe = get_all_axe([grad_exps, shap_exps, lime_exps, ig_exps], x_test_df, y_pred)
    print(f'{grad_axe[141]=}')
    print(f'{shap_axe[141]=}')
    print(f'{lime_axe[141]=}')
    print(f'{ig_axe[141]=}')
    different_indices = []
    for i, (grad_val, shap_val, lime_val, ig_val) in enumerate(zip(grad_axe, shap_axe, lime_axe, ig_axe)):
        if len({grad_val, shap_val, lime_val, ig_val}) == 4:
            different_indices.append(i)
    
    print(different_indices)

    plot_all(model, x_train, x_test, feature_names, index=141)


if __name__ == "__main__":
    main()
