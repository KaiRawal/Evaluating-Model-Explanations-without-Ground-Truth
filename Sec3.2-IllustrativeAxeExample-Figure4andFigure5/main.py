import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm



np.random.seed(1)


# Define the centers of the clusters
centers = [(2, 2), (-2, 2), (-2, -2), (2, -2)]

# Generate the dataset

# X, _ = make_blobs(n_samples=[5000, 5000, 5000, 5000], centers=centers, cluster_std=0.2, random_state=0)


cluster_size = 5000
covariances = [np.array([[0.3, 0.1], [0.1, 0.2]]), np.array([[0.2, 0.1], [0.1, 0.3]]),
               np.array([[0.25, 0.05], [0.05, 0.25]]), np.array([[0.15, -0.05], [-0.05, 0.15]])]
X = np.vstack([np.random.multivariate_normal(mean=center, cov=cov, size=cluster_size) 
               for center, cov in zip(centers, covariances)])

center1 = centers[0]
covariance1 = covariances[0]

def compute_probability_for_center1(points):
    rv = multivariate_normal(mean=center1, cov=covariance1)
    probabilities = rv.pdf(points)
    return np.mean(probabilities)

def compute_mixture_probability(points, centers=centers, covariances=covariances):
    total_probability = 0
    for point in points:
        point_probability = 0
        for center, cov in zip(centers, covariances):
            rv = multivariate_normal(mean=center, cov=cov)
            point_probability += rv.pdf(point)
        total_probability += point_probability / len(centers)
    return total_probability / len(points)

# Example usage:
# points = [(px1, py1), (px2, py2), ..., (pxn, pyn)]
# probabilities = compute_mixture_probability(points, centers, covariances)
# print(probabilities)

# Assign labels to the clusters
# Clusters at (2, 2) and (2, -2) are labeled 1, others are labeled 0

# Convert X and labels into a DataFrame
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['label'] = np.where(df['x1'] > 0, 1, 0)

# Find the 500 nearest points to x1=2 using absolute differences
x1_target = center1[0]
df['distance_to_x1_target'] = abs(df['x1'] - x1_target)

# Sort by distance and get the 500 nearest points for x1
nearest_points_to_x1_2 = df.nsmallest(9000, 'distance_to_x1_target')[['x1']].values

# Find the 500 nearest points to x2=2 using absolute differences
x2_target = center1[1]
df['distance_to_x2_target'] = abs(df['x2'] - x2_target)

# Sort by distance and get the 500 nearest points for x2
nearest_points_to_x2_2 = df.nsmallest(9000, 'distance_to_x2_target')[['x2']].values

print('======')
print('x1 nearest points:')
print(len(nearest_points_to_x1_2))
print(sum(nearest_points_to_x1_2) / len(nearest_points_to_x1_2))
x1_min = min(nearest_points_to_x1_2)
x1_max = max(nearest_points_to_x1_2)
print(f'{x1_min=}, {x1_max=}')

print('x2 nearest points:')
print(len(nearest_points_to_x2_2))
print(sum(nearest_points_to_x2_2) / len(nearest_points_to_x2_2))
x2_min = min(nearest_points_to_x2_2)
x2_max = max(nearest_points_to_x2_2)
print(f'{x2_min=}, {x2_max=}')



# Plot the data
# plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')  # Adjusted figure size
ax.set_aspect('equal', 'box')

plt.grid(True, linestyle='dotted', alpha=0.5)

# Add subtle highlight to the right half of the scatterplot (x > 0)
plt.axvspan(0, 3.5, color='orange', alpha=0.1)
plt.axvspan(-3.5, 0, color='lightgreen', alpha=0.1)
plt.text(1.75, -3, r'positive ($X_1 > 0)$', fontsize=15, ha='center', va='top', color='darkorange')
plt.text(-1.75, -3, r'negative ($X_1 < 0)$', fontsize=15, ha='center', va='top', color='green')

plt.scatter(df[df['label'] == 0]['x1'], df[df['label'] == 0]['x2'], marker='o', label='positive samples', alpha=0.3, s=10, color='lightgreen')
plt.scatter(df[df['label'] == 1]['x1'], df[df['label'] == 1]['x2'], marker='x', label='negative samples', alpha=0.3, s=10, color='orange')

# Add special markers for points A, B, C, and D
special_points = [(2, 2, 'Q'), (-2, 2, 'R'), (-2, -2, 'S'), (2, -2, 'T')]
for x, y, label in special_points:
    plt.scatter(x, y, marker='D', color='black', s=10)
    plt.text(x-0.1, y-0.3, f'{label}', fontsize=15, ha='right', va='bottom', color='black')

plt.xlabel(r'$X_1$', fontsize=19)
plt.ylabel(r'$X_2$', fontsize=19)
plt.gca().set_aspect('equal', 'box')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

plt.text(2.0, 0.35, r'$\eta_a$', fontsize=19, ha='center', va='center', color='red', fontweight='bold')
plt.annotate(
    '', 
    xy=(1.5, 0.1), 
    xytext=(2.5, 0.1),
    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
)

plt.text(-0.4, 2, r'$\Delta_b$', fontsize=19, ha='center', va='center', color='blue', fontweight='bold')
plt.annotate(
    '', 
    xy=(-0.1, 2.5), 
    xytext=(-0.1, 1.5),
    arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5)
)

# Set the limits for both axes
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)

# Reduce the frequency of xticks on both axes
plt.xticks(np.arange(-3, 4, 2))
plt.yticks(np.arange(-3, 4, 2))

plt.savefig('scatter.pdf', dpi=300)

plt.cla()
plt.clf()
plt.close()

# print(df)

class SimpleModel:
    def predict(self, X):
        return np.where(X[:, 0] > 0, 1, 0)
    
    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], 2))
        proba[:, 1] = np.where(X[:, 0] > 0, 1.0, 0.0)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

# Example usage:
# model = SimpleModel()
# predictions = model.predict(X)
# probabilities = model.predict_proba(X)


reference_point_x1, reference_point_x2 = 2, 2

model = SimpleModel()
predictions = model.predict(df[['x1', 'x2']].values)

# Calculate accuracy
accuracy = np.mean(predictions == df['label'].values)
print(f"Accuracy: {accuracy:.2f}")


df_copy_x1 = df.copy(deep=True).drop(columns=['x2'])
df_copy_x1['x1'] = abs(df_copy_x1['x1'] - reference_point_x1)
df_copy_x1 = df_copy_x1.sort_values(by='x1')
df_copy_x1['label_cumsum'] = df_copy_x1['label'].cumsum()
df_copy_x1['label_cumsum_avg'] = df_copy_x1['label_cumsum'] / np.arange(1, len(df_copy_x1) + 1)

cutoff_axe_x = (cluster_size-1) * 2
df_copy_x1 = df_copy_x1.head(cutoff_axe_x)

# print(df_copy_x1)

df_copy_x2 = df.copy(deep=True).drop(columns=['x1'])
df_copy_x2['x2'] = abs(df_copy_x2['x2'] - reference_point_x2)
df_copy_x2 = df_copy_x2.sort_values(by='x2')
df_copy_x2['label_cumsum'] = df_copy_x2['label'].cumsum()
df_copy_x2['label_cumsum_avg'] = df_copy_x2['label_cumsum'] / np.arange(1, len(df_copy_x2) + 1)
# print(df_copy_x2)

df_copy_x2 = df_copy_x2.head(9999)

# print(df_copy_x2['label_cumsum_avg'].tolist())


def calculate_pgi(data_point, perturb_feature, noise_width):
    # Create 100 perturbed samples
    perturbed_samples = np.tile(data_point, (100, 1)).astype(float)
    
    # Add Gaussian noise to the specified feature
    noise = np.random.normal(0, noise_width, 100)
    if perturb_feature == 'x1':
        perturbed_samples[:, 0] += noise
    elif perturb_feature == 'x2':
        perturbed_samples[:, 1] += noise
    else:
        raise ValueError("perturb_feature must be either 'x1' or 'x2'")
    
    # Get the predicted probability for the original data point
    original_proba = model.predict_proba(data_point.reshape(1, -1).astype(float))[0, 1]
    
    # Get the predicted probabilities for the perturbed samples
    perturbed_probas = model.predict_proba(perturbed_samples)[:, 1]
    
    # Calculate the mean absolute difference
    pgi = np.mean(np.abs(perturbed_probas - original_proba))
    
    # Calculate the likelihood of the perturbed samples belonging to gaussian1
    likelihood_gaussian1 = compute_probability_for_center1(perturbed_samples)
    
    # Calculate the overall mixture model probability for the perturbed samples
    mixture_probability = compute_mixture_probability(perturbed_samples)
    
    return float(pgi), likelihood_gaussian1, mixture_probability

# Example usage:
# data_point = np.array([1.5, 2.5])
# pgi_x1 = calculate_pgi(data_point, 'x1', 0.1)
# pgi_x2 = calculate_pgi(data_point, 'x2', 0.1)
# print(f"PGI for x1: {pgi_x1:.4f}")
# print(f"PGI for x2: {pgi_x2:.4f}")

perturb_widths = [round(i, 2) for i in np.arange(0.01, 9.03, 0.005)]

pgi_x1 = []
pgi_x2 = []
pgi_x1_qprobs = []
pgi_x2_qprobs = []
pgi_x1_total_probs = []
pgi_x2_total_probs = []

for nw in tqdm(perturb_widths, desc="Calculating PGI"):
    pgi_ea_x1, pgi_probsQ_x1, pgi_probs_all_x1 = calculate_pgi(np.array([reference_point_x1, reference_point_x2]), 'x1', nw)
    pgi_eb_x2, pgi_probsQ_x2, pgi_probs_all_x2 = calculate_pgi(np.array([reference_point_x2, reference_point_x2]), 'x2', nw)
    pgi_x1.append(pgi_ea_x1)
    pgi_x2.append(pgi_eb_x2)
    pgi_x1_qprobs.append(pgi_probsQ_x1)
    pgi_x2_qprobs.append(pgi_probsQ_x2)
    pgi_x1_total_probs.append(pgi_probs_all_x1)
    pgi_x2_total_probs.append(pgi_probs_all_x2)

# print(pgi_x1)
# print(pgi_x2)



plt.figure(figsize=(10, 6))

plt.plot(perturb_widths, pgi_x2, label=r'$e_b$', color='blue')
plt.plot(perturb_widths, pgi_x1, label=r'$e_a$', color='red', linewidth=0.8)


# plt.plot(perturb_widths, pgi_x2_qprobs, label=r'$e_b - qprob$', color='blue', linestyle='--', linewidth=0.5)
# plt.plot(perturb_widths, pgi_x1_qprobs, label=r'$e_a - qprob$', color='red', linestyle='--', linewidth=0.5)

window_size = 20  # Define the window size for the moving average

# Calculate the moving average for pgi_x2_qprobs
pgi_x2_qprobs_moving_avg = np.convolve(pgi_x2_qprobs, np.ones(window_size)/window_size, mode='valid')
# Calculate the moving average for pgi_x1_qprobs
pgi_x1_qprobs_moving_avg = np.convolve(pgi_x1_qprobs, np.ones(window_size)/window_size, mode='valid')

# Adjust perturb_widths to match the length of the moving averages
adjusted_perturb_widths = perturb_widths[:len(pgi_x2_qprobs_moving_avg)]

plt.plot(adjusted_perturb_widths, pgi_x2_qprobs_moving_avg, label=r'$P(\Delta_{\mathcal{b}} \subseteq \mathcal{M})$', color='blue', linestyle='--', alpha=0.5)
plt.plot(adjusted_perturb_widths, pgi_x1_qprobs_moving_avg, label=r'$P(\Delta_{\mathcal{a}} \subseteq \mathcal{M})$', color='red', linestyle='--', alpha=0.5)


# plt.plot(perturb_widths, pgi_x2_total_probs, label=r'$e_b - overall prob$', color='blue', linestyle=':')
# plt.plot(perturb_widths, pgi_x1_total_probs, label=r'$e_a - overall prob$', color='red', linestyle=':')

plt.xscale('symlog')
plt.xlabel('Perturbation Width', fontsize=27)

plt.ylabel('PGI Explanation Quality &\nProb(On-manifold Perturbation)', fontsize=23)
# plt.title('PGI vs Perturbation Width', fontsize=23)

plt.xticks(ticks=[0, 1, 2, 4, 8], labels=['0', '1', '2', '4', '8'], fontsize=23)  # Ensure the range of x-axis includes 8
plt.yticks(fontsize=23)  # Increased font size for yticks

plt.legend(fontsize=23, title_fontsize=26, loc='upper center')
plt.grid(True)

plt.tight_layout()

plt.savefig('pgi_plot.pdf', dpi=300)
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(df_copy_x2) + 1), df_copy_x2['label_cumsum_avg'], label=r'$e_b$', color='blue')
plt.plot(np.arange(1, len(df_copy_x1) + 1), df_copy_x1['label_cumsum_avg'], label=r'$e_a$', color='red')
plt.xscale('symlog')
plt.xlabel('Number of Nearest Neighbours', fontsize=27)
plt.ylabel('AXE Explanation Quality', fontsize=23)
# plt.title('Cumulative Sum Avg vs Index', fontsize=23)
plt.xticks(fontsize=23)  # Increased font size for xticks
plt.yticks(fontsize=23)  # Increased font size for yticks
plt.legend(fontsize=23, title_fontsize=26)
plt.grid(True)

plt.tight_layout()

plt.savefig('axe_plot.pdf', dpi=300)
# plt.show()





