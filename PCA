@author: ronay
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############################# Making the A array ##############################
A = []

files   = ['Hours_worked_per_week', 'Employment_rate', 'Unemployment_rate', 'GDP_per_capita', 'Labour_mobility', 'Working_population']
countries = ['Belgium','Greece','Lithuania','Portugal','Bulgaria','Spain','Luxembourg','Romania','Czechia','France','Hungary','Slovenia','Denmark','Croatia','Slovakia','Germany','Italy','Netherlands','Finland','Estonia','Cyprus','Austria','Sweden','Ireland','Latvia','Poland','Switzerland','Norway','Iceland']
symbol  = ['BE','EL','LT','PT','BG','ES','LU','RO','CZ','FR','HU','SI','DK','HR','SK','DE','IT','NL','FI','EE','CY','AT','SE','IE','LV','PL','CH','NO','IS']

A.append(files)

def filesort(file):
    with open(file, "r") as f:
        for line in f:
            country, value = line.split()
            value = float(value)
            
            found = False
            
            for row in A:
                if row[0] == country:
                    row.append(value)
                    found = True
                    break
            if not found:
                n = countries.index(country)
                symb = symbol[n]
                hrs = [country, symb, value]  
                A.append(hrs)

filesort(r"Hours_worked_per_week")
filesort(r"Employment_rate")
filesort(r"Unemployment_rate")
filesort(r"GDP_per_capita")
filesort(r"Labour_mobility")
filesort(r"Working_population")

###################### Plotting the unaltered A matrix   ######################

files_order =  ['Employment_rate', 'Unemployment_rate', 'GDP_per_capita', 'Labour_mobility', 'Working_population', 'Hours_worked_per_week']
reorder_indices = [files.index(metric) for metric in files_order]
files = files_order

countries = [row[0] for row in A]
countries = countries[1:]
A = A[1:]
A = [row[2:]for row in A]
A = np.array(A, dtype=float)
A = A[:, reorder_indices]
A = np.transpose(A)


fig, axs = plt.subplots(2, 3, figsize=(40, 12))  # 2 rows, 3 columns of plots
axs = axs.flatten()  

for i in range(len(files)):
    metric_name = files[i]
    values = A[i]
    ax = axs[i]
    ax.bar(countries, values, color='darkmagenta')
    ax.set_title(metric_name.replace('_', ' '), fontsize=40)  
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=90, fontsize=24)  
    ax.set_ylabel(metric_name.replace('_', ' '), fontsize=26) 
    ax.tick_params(axis='y', labelsize=24)

plt.tight_layout()
plt.show()

######################## Making mean = 0 ###################################### 

means = np.mean(A, axis=1)
mean_zero = A - means[:, np.newaxis]
 
fig, axs = plt.subplots(2, 3, figsize=(40, 12))  # 2 rows, 3 columns of plots
axs = axs.flatten() 

for i in range(len(files)):
    metric_name = files[i]
    values = mean_zero[i]
    ax = axs[i]
    ax.bar(countries, values, color = 'tomato')
    ax.set_title(metric_name.replace('_', ' '), fontsize=40)  
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=90, fontsize=24)  
    ax.set_ylabel(metric_name.replace('_', ' '), fontsize=26) 
    ax.tick_params(axis='y', labelsize=24)
plt.tight_layout()
plt.show()

#################### Normalising the values so the norms sum to 1 #############


def normalise_rows(A):
    norms = np.linalg.norm(A, axis = 1, keepdims = True)
    A_norm = 1/norms
    D = np.diagflat(A_norm)
    return D

D = normalise_rows(mean_zero)

DA = D@mean_zero

print(np.linalg.norm((DA), axis=1)) 

fig, axs = plt.subplots(2, 3, figsize=(40, 12))  # 2 rows, 3 columns of plots
axs = axs.flatten() 

for i in range(len(files)):
    metric_name = files[i]
    values = DA[i]
    ax = axs[i]
    ax.bar(countries, values, color = 'darkslategrey')
    ax.set_title(metric_name.replace('_', ' '), fontsize=40)  
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, rotation=90, fontsize=24)  
    ax.set_ylabel(metric_name.replace('_', ' '), fontsize=26) 
    ax.tick_params(axis='y', labelsize=24)
plt.tight_layout()
plt.show()

################## Question 5 - Correlation Matrix ############################

DAT = np.transpose(DA)

C = DA @ DAT

plt.figure(figsize=(12, 10))

# Create heatmap using imshow
heatmap = plt.imshow(C, cmap='cool', vmin=-1, vmax=1, interpolation='nearest')
# Add labels and title
plt.xticks(np.arange(len(files)), [name.replace('_', '\n') for name in files], rotation=0)
plt.yticks(np.arange(len(files)), files)
plt.xlabel("Metrics")
plt.ylabel("Metrics")
plt.title("Covariance Matrix Heatmap")
cbar = plt.colorbar(heatmap, shrink=0.8)
cbar.set_label("Covariance Value")
for i in range(len(files)):
    for j in range(len(files)):
        text = plt.text(j, i, f"{C[i,j]:.2f}",
                       ha="center", va="center", color="w", fontsize=8)

plt.tight_layout()
plt.show()

evals, evecs = np.linalg.eigh(C)
######################## Question 6 - PCA #####################################

idx = np.argsort(evals)[::-1]

evals = evals[idx]
evecs = evecs[:, idx]

pc1 = evecs[:, 0]
pc2 = evecs[:, 1]

X = DA.T  
projections = np.column_stack([
    X @ pc1,  # First principal component scores
    X @ pc2   # Second principal component scores
])

plt.figure(figsize=(10,7))
plt.scatter(projections[:, 0], projections[:, 1], alpha=0.7)

# Add country labels using symbols
for i, sym in enumerate(symbol):
    plt.text(projections[i, 0], projections[i, 1], sym, 
             ha='center', va='bottom', fontsize=8)

plt.xlabel(f'PC1 ({evals[0]/np.sum(evals):.1%} variance)')
plt.ylabel(f'PC2 ({evals[1]/np.sum(evals):.1%} variance)')
plt.title('Manual PCA Projection Using Correlation Matrix')
plt.grid(True)
plt.show()

# Print explained variance
print("Eigenvalues:", evals)
print(f"Total explained variance by first 2 PCs: "
      f"{(evals[0] + evals[1])/np.sum(evals):.1%}")


############################# Question 9 ######################################

A_prime = DA[:5, :].T 
b = DA[5, :].T 

def ridge_regression(A, b, lam):
    n = A.shape[1]
    I = np.eye(n)
    return np.linalg.inv(A.T @ A + lam * I) @ A.T @ b

lambdas = [10000000.0, 10000.0, 10.0, 0.0]
results = {}

for lam in lambdas:
    x = ridge_regression(A_prime, b, lam)
    results[lam] = x



df_ridge = pd.DataFrame(results).T
df_ridge.columns = files[:5]
print(df_ridge)

lambda_chosen = 10.0

x = results[lambda_chosen]

predicted_hours = A_prime @ x

for country, prediction in zip(countries, predicted_hours):
    print(f"{country:15s}: {prediction:.3f}")
    
    
for lam, x in results.items():
    predicted = A_prime @ x

    plt.figure(figsize=(10, 7))
    plt.scatter(b, predicted, color='royalblue')
    for i, sym in enumerate(symbol):
        plt.text(b[i], predicted[i], sym, fontsize=8, ha='center', va='bottom')

    plt.title(f'Actual vs. Predicted Hours Worked (λ = {lam})')
    plt.xlabel('Actual Hours Worked')
    plt.ylabel('Predicted Hours Worked')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
