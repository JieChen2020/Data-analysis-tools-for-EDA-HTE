import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Load the csv file containing the data
file_path = 'Reaction kinetic data 60.csv'
data = pd.read_csv(file_path)

# Extract the values from the DataFrame
time = data['time'].to_numpy()
concentration_c = data['concentration_c'].to_numpy()
concentration_a = data['concentration_a'].to_numpy()
initial_concentration_b = 2 * concentration_a[0]
print(concentration_c)


# Define differential equations for the four kinetic models
def model_r1a(y, t, k):
    a = y[0]
    return [-k * a]


def model_r1ab(y, t, k):
    a, b = y
    return [-k * a * b, -k * a * b]


def model_r1a2(y, t, k):
    a = y[0]
    return [-k * a ** 2]


def model_r1ab2(y, t, k):
    a, b = y
    return [-k * a * b ** 2, -k * a * b ** 2]


# Fit function for model r=k[a]
def fit_r1a(t, k):
    y0 = [concentration_a[0]]
    sol = odeint(model_r1a, y0, t, args=(k,))
    return concentration_a[0] - sol[:, 0]


# Fit function for model r=k[a][b]
def fit_r1ab(t, k):
    y0 = [concentration_a[0], initial_concentration_b]
    sol = odeint(model_r1ab, y0, t, args=(k,))
    return concentration_a[0] - sol[:, 0]


# Fit function for model r=k[a]^2
def fit_r1a2(t, k):
    y0 = [concentration_a[0]]
    sol = odeint(model_r1a2, y0, t, args=(k,))
    return concentration_a[0] - sol[:, 0]


# Fit function for model r=k[a][b]^2
def fit_r1ab2(t, k):
    y0 = [concentration_a[0], initial_concentration_b]
    sol = odeint(model_r1ab2, y0, t, args=(k,))
    return concentration_a[0] - sol[:, 0]


# Curve fitting for each model
popt_r1a, pcov_r1a = curve_fit(fit_r1a, time, concentration_c)
popt_r1ab, pcov_r1ab = curve_fit(fit_r1ab, time, concentration_c)
popt_r1a2, pcov_r1a2 = curve_fit(fit_r1a2, time, concentration_c)
popt_r1ab2, pcov_r1ab2 = curve_fit(fit_r1ab2, time, concentration_c)


# Calculate R-squared values
def calculate_r_squared(model_fit, popt):
    residuals = concentration_c - model_fit(time, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((concentration_c - np.mean(concentration_c)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


r2_r1a = calculate_r_squared(fit_r1a, popt_r1a)
r2_r1ab = calculate_r_squared(fit_r1ab, popt_r1ab)
r2_r1a2 = calculate_r_squared(fit_r1a2, popt_r1a2)
r2_r1ab2 = calculate_r_squared(fit_r1ab2, popt_r1ab2)

# Prepare the results
results = pd.DataFrame({
    'Model': ['r=k[a]', 'r=k[a][b]', 'r=k[a]^2', 'r=k[a][b]^2'],
    'k': [popt_r1a[0], popt_r1ab[0], popt_r1a2[0], popt_r1ab2[0]],
    'R-squared': [r2_r1a, r2_r1ab, r2_r1a2, r2_r1ab2]})

print(results)

# plt.figure(figsize=(10, 6))
#
#
# plt.scatter(time, concentration_a, color='black', label='Experimental Data')
#
#
# t_fit = np.linspace(0, 6, 500)
#
#
# plt.plot(t_fit, odeint(model_r1a, [concentration_a[0]], t_fit, args=(popt_r1a[0],))[:, 0], label='Fit: r=k[a]', color='blue')
# plt.plot(t_fit, odeint(model_r1ab, [concentration_a[0], initial_concentration_b], t_fit, args=(popt_r1ab[0],))[:, 0], label='Fit: r=k[a][b]', color='green')
# plt.plot(t_fit, odeint(model_r1a2, [concentration_a[0]], t_fit, args=(popt_r1a2[0],))[:, 0], label='Fit: r=k[a]^2', color='red')
# plt.plot(t_fit, odeint(model_r1ab2, [concentration_a[0], initial_concentration_b], t_fit, args=(popt_r1ab2[0],))[:, 0], label='Fit: r=k[a][b]^2', color='purple')
#
#
# plt.xlabel('Time')
# plt.ylabel('Concentration of C')
# plt.title('Fitting Models to Experimental Data')
# plt.legend()
# plt.grid(True)
#
#
# plt.show()
