# %% [markdown]
# # **TP 1**
# ## *Effets biologiques des rayonnements. Relation entre la dose et la survie cellulaire*
# 
# **Contacts :**
# 
# - Verónica Sorgato: [veronica.sorgato88@gmail.com](mailto:veronica.sorgato88@gmail.com)
# - Samy Kefs: [samy.kefs@inserm.fr](mailto:samy.kefs@inserm.fr)
# - Yannick Arnoud: [yannick.arnoud@lpsc.in2p3.fr](mailto:yannick.arnoud@lpsc.in2p3.fr)
# 
# ---
# ## **Context**
# 
# La  publication  qui  vous  a  été  fournie  s’interesse  à  deux  lignées  de  cellules  lymphatiques humaines :  WSU-DLCL2  et  SK-DHL2B.  Ces  cellules  ont été  soumises  à  deux  types  d’irradiation. 
# D’une part des **photons** (cobalt 60) et d’autre part des **neutrons** rapides. 
# 
# ---
# 
# 

# %% [markdown]
# ---
# ## **Question 1**
# 
#  1. Expliquez ce qu'est une courbe de survie.
# 
#  ---

# %% [markdown]
# ---
# ## **Question 2** 
# 
# 2. Récupérer  graphiquement  les  valeurs  des  mesures {dose, survie et incertitude sur la survie} et les sauvegarder au format excel.
# ---

# %% [markdown]
# ---
# ## **Question 3**
# 
# 3. Ouvrir dans python les données excel
# ---

# %%
# Import the required libraries

import pandas as pd 
import numpy as np


# %%
# Define the path to the XLSX file created earlier

pathToXLSX= "/Users/samy/Desktop/PhD/TP_Dosimetrie_M2/TP1/Data.xlsx"

# %%
# Open the XLSX file and print the sheet names 

myXlsxFile = pd.ExcelFile(pathToXLSX)
print(myXlsxFile.sheet_names)


# %%
# Load each sheet into a list

data_sheets = []
for sheet_name in myXlsxFile.sheet_names:
    
    data = myXlsxFile.parse(sheet_name)
    data_sheets.append(data)
    
data_sheets[1]
    

# %%
data_sheets[0]

# %%
# You can also manually open each sheet

Photon_Sheet= pd.read_excel(pathToXLSX, sheet_name="Sheet2")

Neutron_Sheet= pd.read_excel(pathToXLSX, sheet_name="Sheet1")



# %% [markdown]
# ## **Reminder**
# 
# Pandas returns DataFrames, which are highly practical and powerful for handling Excel files. However, DataFrames can be complex to manipulate and may lead to longer processing times, especially when dealing with large datasets that consume significant memory. 
# That's why, in this case, I am converting the data into NumPy arrays.
# 

# %%
# Convert the dataFrame to numpy array 
photon_array = Photon_Sheet.to_numpy()
neutron_array = Neutron_Sheet.to_numpy()

photon_array




# %% [markdown]
# ## **Reminder**
# 
# Data management and code readability are crucial aspects to avoid errors and, most importantly, to enable others to understand your code.
# 
# It all depends on your needs, but using dictionaries can be very practical and efficient for data management.
# 

# %%
# It could be relevant to use different array with different names 

photon_Dose = photon_array[:, 0]
neutron_Dose = neutron_array[:, 0]

photon_SKDHL2B = photon_array[:,4]
photon_SKDHL2B_upper = photon_array[:, 5]
photon_SKDHL2B_lower = photon_array[:, 6]

neutron_SKDHL2B = neutron_array[:,4]
neutron_SKDHL2B_upper = neutron_array[:, 5]
neutron_SKDHL2B_lower = neutron_array[:, 6]

photon_array = photon_array[~(np.isnan(photon_array).any(axis=1))]
photon_Dose_WSU_DLCL2 = photon_array[:, 0]
photon_WSU_DLCL2 = photon_array[:, 1]
photon_WSU_DLCL2_upper = photon_array[:, 2]
photon_WSU_DLCL2_lower = photon_array[:, 3]

neutron_WSU_DLCL2 = neutron_array[:, 1]
neutron_WSU_DLCL2_upper = neutron_array[:, 2]
neutron_WSU_DLCL2_lower = neutron_array[:, 3]

std_photon_WSU_DLCL2 = ((photon_WSU_DLCL2 + photon_WSU_DLCL2_upper)/4) - ((photon_WSU_DLCL2-photon_WSU_DLCL2_lower)/4)
std_neutron_WSU_DLCL2 = ((neutron_WSU_DLCL2 + neutron_WSU_DLCL2_upper) - (neutron_WSU_DLCL2-neutron_WSU_DLCL2_lower))/(4*np.sqrt(3))
std_photon_SKDHL2B = ((photon_SKDHL2B + photon_SKDHL2B_upper) - (photon_SKDHL2B-photon_SKDHL2B_lower))/(4*np.sqrt(3))
std_neutron_SKDHL2B = ((neutron_SKDHL2B + neutron_SKDHL2B_upper) - (neutron_SKDHL2B-neutron_SKDHL2B_lower))/(4*np.sqrt(3))
std_photon_WSU_DLCL2


# %%
# Best practice to do creat a dictionnary with all data 

dict_Photon = {}
dict_Neutron = {}

for column_names_P in Photon_Sheet.columns:
    dict_Photon[column_names_P]= Photon_Sheet[column_names_P].to_numpy()

for column_names_N in Neutron_Sheet.columns:
    dict_Neutron[column_names_N]= Neutron_Sheet[column_names_N].to_numpy()
    
dict_Neutron

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

# Plot the first scatter plot
plt.scatter(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2, label="Photons", marker='o', s=50, alpha=0.7)

# Plot the second scatter plot
plt.scatter(neutron_Dose, neutron_WSU_DLCL2, label="Neutrons", marker='x', s=50, alpha=0.7)

# Add error bars
plt.errorbar(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2,
             yerr=(photon_WSU_DLCL2_lower, photon_WSU_DLCL2_upper),
             fmt='none', color='blue', alpha=0.5, label="Error bar (Photons)")

plt.errorbar(neutron_Dose, neutron_WSU_DLCL2,
             yerr=(neutron_WSU_DLCL2_lower, neutron_WSU_DLCL2_upper),
             fmt='none', color='orange', alpha=0.5, label="Error bar (Neutrons)")

# Define axis labels
plt.xlabel("Doses (Gy)")
plt.ylabel("Surviving Fraction")

# Set the y-axis scale to logarithmic
plt.yscale('log')

# Graph title
plt.title("Surviving Fraction of WSU_DLCL2 Cells as a Function of Photon and Neutron Doses")

# Add a legend
plt.grid(True)
plt.legend()
plt.show()

# %%
# Do the same thing just using dict 
plt.figure(figsize=(8, 5))

# Plot the first scatter 
plt.scatter(dict_Photon["Doses proton (Gy)"], dict_Photon["WSU_DLCL2"], label="Photon_WSU_DLCL2", marker='o', s=50, alpha=0.7)

# Plot the second one 
plt.scatter(dict_Neutron["Doses Neutron (Gy)"], dict_Neutron["WSU_DLCL2"], label="Neutron_WSU_DLCL2", marker='x', s=50, alpha=0.7)

# add some stuff
plt.xlabel("Doses (Gy)")
plt.ylabel("Surviving fraction")
# put the y scale in log
plt.yscale('log')
plt.title("WSU_DLCL2 in function of photon and neutron doses using dict")


# log scale
plt.yscale('log')
plt.grid(True)
# add a legend
plt.legend()
plt.show()

# %%
plt.figure(figsize=(8, 5))

# Plot the first scatter plot
plt.scatter(photon_Dose, photon_SKDHL2B, label='Photons', marker='o', s=50, alpha=0.7)

# Plot the second scatter plot
plt.scatter(neutron_Dose, neutron_SKDHL2B, label='Neutrons', marker='x', s=50, alpha=0.7)


plt.errorbar(photon_Dose, photon_SKDHL2B,
             yerr=(photon_SKDHL2B_lower, photon_SKDHL2B_upper), fmt='none', color='blue', alpha=0.5,
             label="Error bar (Photons)")

plt.errorbar(neutron_Dose, neutron_SKDHL2B,
             yerr=(neutron_SKDHL2B_lower, neutron_SKDHL2B_upper), fmt='none', color='orange', alpha=0.5,
             label="Error bar (Neutrons)")

plt.yscale('log')


plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving Fraction')


plt.title('Surviving Fraction of SK2B Cells as a Function of Photon and Neutron Doses')


plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ---
# ## **Question 4**
# 
# 4. Representé graphiquement la survie des deux types de cellules pour l'irradiation par faisceaux de photon.
# ---

# %%
plt.figure(figsize=(8, 5))

# Plot the first scatter plot (SK2B)
plt.scatter(photon_Dose, photon_SKDHL2B, label='SK2B', marker='o', s=50, alpha=0.7)

# Plot the second scatter plot (WSU_DLCL2)
plt.scatter(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2, label='WSU_DLCL2', marker='x', s=50, alpha=0.7)


plt.errorbar(photon_Dose, photon_SKDHL2B,
             yerr=(photon_SKDHL2B_lower, photon_SKDHL2B_upper),
             fmt='none', color='blue', alpha=0.5)


plt.errorbar(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2,
             yerr=(photon_WSU_DLCL2_lower, photon_WSU_DLCL2_upper),
             fmt='none', color='orange', alpha=0.5)


plt.yscale('log')


plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving Fraction')


plt.title('Surviving Fraction of SK2B and WSU_DLCL2 Cells in Function of Photon Doses')


plt.legend()
plt.grid(True)
plt.show()

# %%
#Do the same thing but the changing the variable in scatter plot function for neutron 

plt.figure(figsize=(8, 5))

# Plot the first scatter plot (SK2B)
plt.scatter(neutron_Dose, neutron_SKDHL2B, label='SK2B', marker='x', s=50, alpha=0.7)

# Plot the second scatter plot (WSU_DLCL2)
plt.scatter(neutron_Dose, neutron_WSU_DLCL2, label="WSU_DLCL2", marker='x', s=50, alpha=0.7)


plt.errorbar(neutron_Dose, neutron_SKDHL2B,
             yerr=(neutron_SKDHL2B_lower, neutron_SKDHL2B_upper),
             fmt='none', color='orange', alpha=0.5)


plt.errorbar(neutron_Dose, neutron_WSU_DLCL2,
             yerr=(neutron_WSU_DLCL2_lower, neutron_WSU_DLCL2_upper),
             fmt='none', color='blue', alpha=0.5)


plt.yscale('log')


plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving Fraction')


plt.title('Surviving Fraction of SK2B and WSU_DLCL2 Cells in Function of Neutron Doses')


plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ---
# ## Data Trend Modeling
# 
# We will model the data trend using the quadratic linear model: 
# 
# 
# $$S = e^{-\alpha \cdot D - \beta \cdot D^2}$$
# 
# 
# There are two approaches available: fitting directly with the two-parameter decreasing exponential function or fitting the natural logarithm of survival with a second-order polynomial.
# 

# %% [markdown]
# 
# ## **Question 6**
# 
# 6. Ajuster les courbes de survie avec les 2 approches et affichez les graphiquement
# ---

# %%
from scipy.optimize import curve_fit, least_squares

# %%
# You have to definie the model for the fitting 
# in our case we have to fit whith two different model 

def quadratic_model(Dose,a,b):
    return -(a*Dose)-(b*(Dose**2))

def exponential_model(Dose,a,b):
    return np.exp(-((a*Dose)+(b*(Dose**2))))



# %%
a = np.arange(0, 5, 1)
print(a)
b = np.copy(a)
print(b) 
b[4] = 0
print(b)
print(a)

# %% [markdown]
# ### Passage en log des differentes matrices 
# ### Calcule des differents fit a réaliser 

# %%
# Now we have to fit our data, first with the quadratic model 

# Transforming data to logarithmic scale
photon_SKDHL2B_log = np.log(np.copy(photon_SKDHL2B))
std_photon_SKDHL2B_log = np.log(np.copy(std_photon_SKDHL2B))
photon_WSU_DLCL2_log = np.log(np.copy(photon_WSU_DLCL2))
std_photon_WSU_DLCL2_log = np.log(std_photon_WSU_DLCL2)

neutron_SKDHL2B_log = np.log(np.copy(neutron_SKDHL2B))
std_neutron_SKDHL2B_log = np.log(np.copy(std_neutron_SKDHL2B))
neutron_WSU_DLCL2_log = np.log(np.copy(neutron_WSU_DLCL2))
std_neutron_WSU_DLCL2_log = np.log(np.copy(std_neutron_WSU_DLCL2))

# Fitting the data with the quadratic model using curve_fit

parameters_Quad_Photon_WSU_DLCL2, cov_Photon_Quad_WSU_DLCL2 = curve_fit(quadratic_model,
                                                                        photon_Dose_WSU_DLCL2,photon_WSU_DLCL2_log,
                                                                        sigma=std_photon_WSU_DLCL2_log,
                                                                        bounds=([0,0],[np.inf,np.inf])
                                                                        )
print(parameters_Quad_Photon_WSU_DLCL2)

parameters_Quad_Photon_SK2B, cov_Photon_Quad_SK2B = curve_fit(quadratic_model,
                                                              photon_Dose,photon_SKDHL2B_log,
                                                              sigma=std_photon_SKDHL2B_log,
                                                              bounds=([0,0],[np.inf,np.inf]))

parameters_Quad_Neutron_WSU_DLCL2, cov_Quad_Neutron_WSU_DLCL2 = curve_fit(quadratic_model,
                                                                          neutron_Dose,neutron_WSU_DLCL2_log,
                                                                          sigma=std_neutron_WSU_DLCL2_log,
                                                                          bounds=([0,0],[np.inf,np.inf])
                                                                          )

parameters_Quad_Neutron_SK2B, cov_Quad_Neutron_SK2B = curve_fit(quadratic_model,
                                                                neutron_Dose,neutron_SKDHL2B_log,
                                                                sigma=std_neutron_SKDHL2B_log,
                                                                bounds=([0,0],[np.inf,np.inf]))
print(parameters_Quad_Neutron_SK2B)
                                                               
                                                                



print(parameters_Quad_Photon_SK2B)

# %%
# Create a matrix 'm' by stacking two arrays along the last axis
m = np.stack((-photon_Dose_WSU_DLCL2, -photon_Dose_WSU_DLCL2**2), axis=-1)
# Perform a least-squares linear regression using 'm' as the design matrix
result = np.linalg.lstsq(m, photon_WSU_DLCL2_log)
# Print the parameters
print(result)

# %%
# measure of how well the quadratic model fits the 'photon_WSU_DLCL2_log' data
((photon_WSU_DLCL2_log - quadratic_model(photon_Dose_WSU_DLCL2, *parameters_Quad_Photon_WSU_DLCL2))**2).sum()

# %%
%timeit np.linalg.lstsq(m, photon_WSU_DLCL2_log)
%timeit parameters_Quad_Photon_WSU_DLCL2, cov_Photon_Quad_WSU_DLCL2 = curve_fit(quadratic_model, photon_Dose_WSU_DLCL2,photon_WSU_DLCL2_log)


# %%
# Plot it 

plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.scatter(neutron_Dose, neutron_SKDHL2B_log, label='neutron_SK2B', marker='o', s=50, alpha=0.7)
plt.scatter(photon_Dose, photon_SKDHL2B_log, label='photon_SK2B', marker='x', s=50, alpha=0.7)


#plt.yscale('log')
plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving fraction')
plt.title('SK2B Survival Fraction vs Neutron and Photon Doses (log scale)')

x_range_photon = np.linspace(np.min(photon_Dose), np.max(photon_Dose), 100)
x_range_neutron = np.linspace(np.min(neutron_Dose),np.max(neutron_Dose),100)
plt.plot(x_range_neutron , quadratic_model(x_range_neutron , *parameters_Quad_Neutron_SK2B), label='Fit neutron_SK2B', color='blue')
plt.plot(x_range_photon, quadratic_model(x_range_photon, *parameters_Quad_Photon_SK2B), label='Fit photon_SK2B', color='orange')

# Fit equation for SK2B - Neutrons
fit_eq_neutron_SK2B = f"Fit (Neutron_SK2B): {parameters_Quad_Neutron_SK2B[0]:.4f}x + {parameters_Quad_Neutron_SK2B[1]:.4f}x^2"
plt.text(0.01, 0.01, fit_eq_neutron_SK2B, transform=plt.gca().transAxes, fontsize=12, color='blue')

# Fit equation for SK2B - Photons
fit_eq_photon_SK2B = f"Fit (Photon_SK2B):  {parameters_Quad_Photon_SK2B[0]:.4f}x + {parameters_Quad_Photon_SK2B[1]:.4f}x^2"
plt.text(0.01, 0.05, fit_eq_photon_SK2B, transform=plt.gca().transAxes, fontsize=12, color='orange')


plt.errorbar(neutron_Dose, neutron_SKDHL2B_log,
             yerr=(np.abs(std_neutron_SKDHL2B_log)),fmt='none',color='blue',alpha=0.5)
plt.errorbar(photon_Dose, photon_SKDHL2B_log,
             yerr=(np.abs(std_photon_SKDHL2B_log)),fmt='none', color='orange',alpha=0.5)

plt.legend()


## Next PLOT ####


plt.subplot(1,2,2)
plt.scatter(neutron_Dose, neutron_WSU_DLCL2_log, label='neutron_WSU_DLCL2', marker='o', s=50, alpha=0.7)
plt.scatter(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2_log, label='photon_WSU_DLCL2', marker='x', s=50, alpha=0.7)


#plt.yscale('log')
plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving fraction')
plt.title('WSU_DLCL2 Survival Fraction vs Neutron and Photon Doses (log scale)')

plt.plot(x_range_neutron , quadratic_model(x_range_neutron , *parameters_Quad_Neutron_WSU_DLCL2), label='Fit neutron_WSU_DLCL2', color='blue')
plt.plot(x_range_photon, quadratic_model(x_range_photon, *parameters_Quad_Photon_WSU_DLCL2), label='Fit photon_WSU_DLCL2', color='orange')

# Fit equation for WSU_DLCL2 - Neutrons
fit_eq_neutron_WSU_DLCL2 = f"Fit (Neutron_WSU_DLCL2): {parameters_Quad_Neutron_WSU_DLCL2[0]:.4f}x + {parameters_Quad_Neutron_WSU_DLCL2[1]:.4f}x^2"
plt.text(0.01, 0.01, fit_eq_neutron_WSU_DLCL2, transform=plt.gca().transAxes, fontsize=12, color='blue')

# Fit equation for WSU_DLCL2 - Photons
fit_eq_photon_WSU_DLCL2 = f"Fit (Photon_WSU_DLCL2): {parameters_Quad_Photon_WSU_DLCL2[0]:.4f}x + {parameters_Quad_Photon_WSU_DLCL2[1]:.4f}x^2"
plt.text(0.01, 0.06, fit_eq_photon_WSU_DLCL2, transform=plt.gca().transAxes, fontsize=12, color='orange')


plt.errorbar(neutron_Dose, neutron_WSU_DLCL2_log,
             yerr=(np.abs(std_neutron_WSU_DLCL2_log)),fmt='none',color='blue',alpha=0.5)
plt.errorbar(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2_log,
             yerr=(np.abs(std_photon_WSU_DLCL2_log)),fmt='none', color='orange',alpha=0.5)

plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Calcule des fits exponentiels cette fois

# %%
# Now with fit with an exponential model 
print(std_photon_WSU_DLCL2)
print(std_photon_WSU_DLCL2 + photon_WSU_DLCL2)
std_photon_WSU_DLCL2[0]=0.00000001
std_photon_SKDHL2B[0]=0.00000001
std_neutron_WSU_DLCL2[0]=0.00000001
std_neutron_SKDHL2B[0]=0.00000001
parameters_exp_Photon_WSU_DLCL2, cov_Photon_exp_WSU_DLCL2 = curve_fit(exponential_model,
                                                                        photon_Dose_WSU_DLCL2,photon_WSU_DLCL2,
                                                                        sigma=std_photon_WSU_DLCL2,
                                                                        bounds=([0,0],[2,2]))
                                                                        

parameters_exp_Photon_SK2B, cov_Photon_exp_SK2B = curve_fit(exponential_model,
                                                              photon_Dose,photon_SKDHL2B,
                                                              sigma=std_photon_SKDHL2B,
                                                              bounds=([0,0],[np.inf,np.inf]))
                                                              

parameters_exp_Neutron_WSU_DLCL2, cov_exp_Neutron_WSU_DLCL2 = curve_fit(exponential_model,
                                                                          neutron_Dose,neutron_WSU_DLCL2,
                                                                          sigma=std_neutron_WSU_DLCL2,
                                                                          bounds=([0,0],[np.inf,np.inf]))                                                                          

parameters_exp_Neutron_SK2B, cov_exp_Neutron_SK2B = curve_fit(exponential_model,
                                                              neutron_Dose,neutron_SKDHL2B,
                                                              sigma=std_neutron_SKDHL2B,
                                                              bounds=([0,0],[np.inf,np.inf]))


parameters_exp_Photon_SK2B,parameters_exp_Photon_WSU_DLCL2, parameters_exp_Neutron_WSU_DLCL2,parameters_exp_Neutron_SK2B,parameters_Quad_Neutron_SK2B,parameters_Quad_Photon_SK2B

# %%
# let see
plt.figure(figsize=(18, 9))
plt.subplot(1,2,1)

plt.scatter(neutron_Dose, neutron_SKDHL2B, label='neutron_SK2B', marker='o', s=50, alpha=0.7)
plt.scatter(photon_Dose, photon_SKDHL2B, label='photon_SK2B', marker='x', s=50, alpha=0.7)

# Fit equations for SK2B
fit_eq_neutron_SK2B = f"Fit (Neutron_SK2B): S = exp(-({parameters_exp_Neutron_SK2B[0]:.4f}D + {parameters_exp_Neutron_SK2B[1]:.4f}D^2))"
fit_eq_photon_SK2B = f"Fit (Photon_SK2B): S = exp(-({parameters_exp_Photon_SK2B[0]:.4f}D + {parameters_exp_Photon_SK2B[1]:.4f}D^2))"

plt.text(0.02, 0.01, fit_eq_neutron_SK2B, transform=plt.gca().transAxes, fontsize=12, color='blue')
plt.text(0.02, 0.06, fit_eq_photon_SK2B, transform=plt.gca().transAxes, fontsize=12, color='orange')


plt.yscale('log')
plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving fraction')
plt.title('SKDHL2B Survival Fraction vs Neutron and Photon Doses')

x_range_photon = np.linspace(np.min(photon_Dose), np.max(photon_Dose), 100)
x_range_neutron = np.linspace(np.min(neutron_Dose),np.max(neutron_Dose),100)
plt.plot(x_range_neutron , exponential_model(x_range_neutron , *parameters_exp_Neutron_SK2B), label='Fit neutron_SK2B', color='blue')
plt.plot(x_range_photon, exponential_model(x_range_photon, *parameters_exp_Photon_SK2B), label='Fit photon_SK2B', color='orange')

plt.legend()





## Next PLOT ####





plt.subplot(1,2,2)
plt.scatter(neutron_Dose, neutron_WSU_DLCL2, label='neutron_WSU_DLCL2', marker='o', s=50, alpha=0.7)
plt.scatter(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2, label='photon_WSU_DLCL2', marker='x', s=50, alpha=0.7)

# Fit equations for WSU_DLCL2
fit_eq_neutron_WSU_DLCL2 = f"Fit (Neutron_WSU_DLCL2): S = exp(-({parameters_exp_Neutron_WSU_DLCL2[0]:.4f}D + {parameters_exp_Neutron_WSU_DLCL2[1]:.4f}D^2))"
fit_eq_photon_WSU_DLCL2 = f"Fit (Photon_WSU_DLCL2): S = exp(-({parameters_exp_Photon_WSU_DLCL2[0]:.4f}D + {parameters_exp_Photon_WSU_DLCL2[1]:.4f}D^2))"

plt.yscale('log')
plt.xlabel('Dose (Gy)')
plt.ylabel('Surviving fraction')
plt.title('WSU_DLCL2 survival fraction in function of neutron and photon doses')

plt.plot(x_range_neutron , exponential_model(x_range_neutron , *parameters_exp_Neutron_WSU_DLCL2), label='Fit neutron_WSU_DLCL2', color='orange')
plt.plot(x_range_photon, exponential_model(x_range_photon, *parameters_exp_Photon_WSU_DLCL2), label='Fit photon_WSU_DLCL2', color='blue')

plt.text(0.02, 0.01, fit_eq_neutron_WSU_DLCL2, transform=plt.gca().transAxes, fontsize=12, color='blue')
plt.text(0.02, 0.05, fit_eq_photon_WSU_DLCL2, transform=plt.gca().transAxes, fontsize=12, color='orange')


plt.errorbar(photon_Dose_WSU_DLCL2, photon_WSU_DLCL2,
             yerr=(photon_WSU_DLCL2_lower,photon_WSU_DLCL2_upper),
             fmt='none', color='blue', alpha=0.5)

plt.errorbar(neutron_Dose, neutron_WSU_DLCL2,
             yerr=(neutron_WSU_DLCL2_lower,neutron_WSU_DLCL2_upper),
             fmt= 'none', color='orange', alpha=0.5)

plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## **Question 7**
#  
#  7. Comparer les paramétres et la precisions obtenue
# 
# ---

# %%
# Define dictionaries to store parameters for different datasets and models

Alpha_exp={
    'Photon WSU_DLCL2': [0.18,parameters_exp_Photon_WSU_DLCL2[0],parameters_Quad_Photon_WSU_DLCL2[0]],
    'Neutron WSU_DLCL2': [1.20,parameters_exp_Neutron_WSU_DLCL2[0],parameters_Quad_Neutron_WSU_DLCL2[0]],
    'Photon SKDHL2B': [0.43,parameters_exp_Photon_SK2B[0],parameters_Quad_Photon_SK2B[0]],
    'Neutron SKDHL2B': [2.43,parameters_exp_Neutron_SK2B[0],parameters_Quad_Neutron_SK2B[0]],
}

Beta_exp={
    'Photon WSU_DLCL2': [0.11,parameters_exp_Photon_WSU_DLCL2[1],parameters_Quad_Photon_WSU_DLCL2[1]],
    'Neutron WSU_DLCL2': [0.22,parameters_exp_Neutron_WSU_DLCL2[1],parameters_Quad_Neutron_WSU_DLCL2[1]],
    'Photon SKDHL2B': [0.05,parameters_exp_Photon_SK2B[1],parameters_Quad_Photon_SK2B[1]],
    'Neutron SKDHL2B': [0,parameters_exp_Neutron_SK2B[1],parameters_Quad_Neutron_SK2B[1]],
}
Alpha_exp,Beta_exp

# %%
import tabulate as tb
def comparison(name,dict):
    result = []
    for keys, values in dict.items():
        result.append([keys,values[0],values[2],values[1],round((values[0]-values[2])/values[0],3),round((values[0]-values[1])/values[0],3)])
    return result

print (' Table for Alpha ')
print(tb.tabulate(comparison('alpha',Alpha_exp), headers=['Cell line','Publication' ,'Quadratic Value','Exponential value', 'Quadratic difference ', 'Exponential difference '], tablefmt='fancy_outline'),'\n\n')
print (' Table for Beta ')
print(tb.tabulate(comparison('beta',Beta_exp), headers=['Cell line','Publication', 'Quadratic Value','Exponential value', 'Quadratic difference ', 'Exponential difference '], tablefmt='fancy_outline'))       

    

# %% [markdown]
# ---
# ## **Question 9** 
# 
# 9. Evaluer la qualité de votre fit avec la valeur du $\chi^{2}$
# 
# #### **Pour rappel :**
# 
# \begin{equation}
# \chi^2 = \sum \frac{{(observed - expected)^2}}{{expected}}
# \end{equation}
# ---

# %%
Photon1={
    
    'Photon SKDHL2B': [parameters_exp_Photon_SK2B[0],parameters_exp_Photon_SK2B[1],parameters_Quad_Photon_SK2B[0],parameters_Quad_Photon_SK2B[1],photon_SKDHL2B],
}
Photon2={
    'Photon WSU_DLCL2': [parameters_exp_Photon_WSU_DLCL2[0],parameters_exp_Photon_WSU_DLCL2[1],parameters_Quad_Photon_WSU_DLCL2[0],parameters_Quad_Photon_WSU_DLCL2[1],photon_WSU_DLCL2],
    
}

Neutron={
    'Neutron WSU_DLCL2': [parameters_exp_Neutron_WSU_DLCL2[0],parameters_exp_Neutron_WSU_DLCL2[1],parameters_Quad_Neutron_WSU_DLCL2[0],parameters_Quad_Neutron_WSU_DLCL2[1],neutron_WSU_DLCL2],
    'Neutron SKDHL2B': [parameters_exp_Neutron_SK2B[0],parameters_exp_Neutron_SK2B[1],parameters_Quad_Neutron_SK2B[0],parameters_Quad_Neutron_SK2B[1],neutron_SKDHL2B],
}

# %%
import tabulate as tb

def chi_sqrt(dose, dict):
    result = []
    for keys,values in dict.items():
        degrees_Lib = dose.shape[0]-2
        
        residual_Quad = quadratic_model(dose,values[2],values[3]) - np.log(values[4])
        chi2_Quad = np.sum(np.nan_to_num((residual_Quad**2)/np.abs((np.log(values[4])))))
        
        
        residual_Exp = (exponential_model(dose,values[0],values[1]))- values[4]
        chi2_Exp = np.sum((residual_Exp**2)/values[4])
        
        chi2_Exp_reduced = chi2_Exp/degrees_Lib
        chi2_Quad_reduced = chi2_Quad/degrees_Lib
        
        
        
        print( f" {keys} Chi2 reduced : Exp = {chi2_Exp_reduced} , Quad : {chi2_Quad_reduced}")
        result.append([keys,chi2_Exp_reduced,chi2_Quad_reduced])
    return result
       
table = chi_sqrt(neutron_Dose,Neutron)

table.append(np.array(chi_sqrt(photon_Dose,Photon1)).flatten())

table.append(np.array(chi_sqrt(photon_Dose_WSU_DLCL2,Photon2)).flatten())


print("\n\n",tb.tabulate(table, headers=['Name', 'Chi2 reduced Exp', 'Chi2 reduced Quad'], tablefmt='fancy_outline'))
        
    
        

# %% [markdown]
# ---
# ## **Question 10** 
# 
# 10. Calculez le rapport 𝛼/𝛽 et l’incertitude associée grâce à la matrice de covariance pour chaque type d’irradiation et de lignée cellulaire.**

# %% [markdown]
# ### **Pour rappel** 
# 
# 
# \begin{equation}
#     \sigma^2_{\frac{\alpha}{\beta}}=\left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\alpha}}\right)^2  \sigma^2_{\alpha}+
#     \left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\alpha}}\right) \left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\beta}}\right) \sigma^2_{\alpha\beta}+
#     \left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\alpha}}\right) \left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\beta}}\right) \sigma^2_{\beta\alpha}+
#     \left(\frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\beta}}\right)^2  \sigma^2_{\beta}
# \end{equation}
# 
#  Dans notre cas nous avons seulement la matrice de covariance
# $$
#     \begin{pmatrix}
#         \sigma^2_{\alpha} & \sigma_{\alpha\beta}\\
#         \sigma_{\beta\alpha} & \sigma^2_{\beta}
#     \end{pmatrix}
# $$
# 
# Nous allons donc utilisé les produits matriciels en utilisant la matrice de covariance 
# 
# \begin{equation}
#     \sigma^2_{\frac{\alpha}{\beta}}= 
#     \begin{pmatrix}
#         \frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\alpha}} & \frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\beta}}
#     \end{pmatrix}
#     \begin{pmatrix}
#         \sigma^2_{\alpha} & \sigma_{\alpha\beta}\\
#         \sigma_{\beta\alpha} & \sigma^2_{\beta}
#     \end{pmatrix}
#     \begin{pmatrix}
#         \frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\alpha}} \\ \frac{\partial_{\frac{\alpha}{\beta}}}{\partial_{\beta}}
#     \end{pmatrix}
# \end{equation}
# 

# %% [markdown]
# Dérivée partielle par rapport à $\alpha$
# 
# $$
# \frac{\partial}{\partial \alpha} \left(\frac{\alpha}{\beta}\right) = \frac{1}{\beta} \cdot \frac{\partial}{\partial \alpha} (\alpha) = \frac{1}{\beta}
# $$
# 
# Dérivée partielle par rapport à $\beta$
# 
# $$
# \frac{\partial}{\partial \beta} \left(\frac{\alpha}{\beta}\right) = \alpha \cdot \frac{\partial}{\partial \beta} \left(\frac{1}{\beta}\right) = -\frac{\alpha}{\beta^2}
# $$
# ---

# %%
param={
    'Photon WSU_DLCL2 exponential': [parameters_exp_Photon_WSU_DLCL2[0],parameters_exp_Photon_WSU_DLCL2[1],cov_Photon_exp_WSU_DLCL2],
    'Photon WSU_DLCL2 quadratic': [parameters_Quad_Photon_WSU_DLCL2[0],parameters_Quad_Photon_WSU_DLCL2[1],cov_Photon_Quad_WSU_DLCL2],
    'Photon SKDHL2B exponential': [parameters_exp_Photon_SK2B[0],parameters_exp_Photon_SK2B[1],cov_Photon_exp_SK2B],
    'Photon SKDHL2B quadratic': [parameters_Quad_Photon_SK2B[0],parameters_Quad_Photon_SK2B[1],cov_Photon_Quad_SK2B],
    'Neutron WSU_DLCL2 exponential': [parameters_exp_Neutron_WSU_DLCL2[0],parameters_exp_Neutron_WSU_DLCL2[1],cov_exp_Neutron_WSU_DLCL2],
    'Neutron WSU_DLCL2 quadratic': [parameters_Quad_Neutron_WSU_DLCL2[0],parameters_Quad_Neutron_WSU_DLCL2[1],cov_Quad_Neutron_SK2B],
    'Neutron SKDHL2B exponential': [parameters_exp_Neutron_SK2B[0],parameters_exp_Neutron_SK2B[1],cov_exp_Neutron_SK2B],
    'Neutron SKDHL2B Quadratic ': [parameters_Quad_Neutron_SK2B[0],parameters_Quad_Neutron_SK2B[1],cov_Quad_Neutron_SK2B],
    }
param


    
    

# %%
def incertitude(dict):
    result=[]
    for keys, values in dict.items():
        
        derivative = []
        derivative.append(1/values[1])
        derivative.append(values[0]/(values[1]**2))
        derivative= np.asarray(derivative)
        std_variance = np.sqrt(np.dot(derivative,np.dot(values[2],derivative.T)))
        result.append([keys,values[0],values[1], values[0]/values[1] ,std_variance])
        print(f'{"α"}/ {"β"} : {values[0]/values[1]}, standard deviation of {keys} : {std_variance}')
    return result
print("\n\n",tb.tabulate(incertitude(param), headers=['Name', 'α', 'β','α/β','Standard deviation'], tablefmt='fancy_outline'))

# %% [markdown]
# ---
# ## **Question 12**
# 
# On considère la lignée cellulaire WSU-DLCL2. À partir des courbes de survie cellulaire obtenues à la question 6, déterminer par interpolation la dose à appliquer avec des photons, puis des neutrons, pour avoir l’effet biologique suivant :
# 
# * 1 % de survie
# * 10 % de survie
# * 50 % de survie
# * 80 % de survie
# * 90 % de survie
# 
# ---
# 
# 

# %%
# for solving the equation there is differents methode 
import math
from scipy import roots

def solve_quadratic_equation(a, b, c):
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    # Check the discriminant
    if discriminant > 0:
        # Two distinct real solutions
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        if x1 >=0 and x2>=0:
            return np.array([x1,x2])
        elif x1 >=0 and x2<=0:
            return np.array([x1])
        else:
            return np.array([x2])
    elif discriminant == 0:
        # One real solution (double root)
        x = -b / (2*a)
        return x,
    else:
        # No real solution (complex solutions)
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(abs(discriminant)) / (2*a)
        solution1 = complex(real_part, imaginary_part)
        solution2 = complex(real_part, -imaginary_part)
        return solution1, solution2



def solver (a , b , c):
    return np.roots([a,b,c])

def solver2(a,b,c):
    return roots([a,b,c])
# Example of usage
a = 1
b = -5
c = 3

solutions = solve_quadratic_equation(a, b, c)

print("Solutions of the equation: ", solutions)
solutions = solver(a, b, c)
print("Solutions of the equation: ", solutions)
solutions = solver2(a, b, c)
print("Solutions of the equation: ", solutions)
%timeit solve_quadratic_equation(a, b, c)
%timeit solver(a, b, c)
%timeit solver2(a, b, c)


# %%
# Now we going to use the models with the alpha and betas value to find that 
# We will use the exponential model 

percentage=[0.01,0.1,0.5,0.8,0.9]  
dose_photon = []
dose_neutron = []
for i in percentage:
    sol_photon = solve_quadratic_equation(param["Photon WSU_DLCL2 exponential"][1],param["Photon WSU_DLCL2 exponential"][0],np.log(i))
    sol_neutron = solve_quadratic_equation(param["Neutron WSU_DLCL2 exponential"][1],param["Neutron WSU_DLCL2 exponential"][0],np.log(i))
    print(f"Dose photon for {i} % of survival = {sol_photon}")
    print(f"Dose neutron for {i} % of survival = {sol_neutron}")
    dose_photon.append(sol_photon)
    dose_neutron.append(sol_neutron)

dose_photon = np.asarray(dose_photon)
dose_neutron = np.asarray(dose_neutron)
percentage = np.asarray(percentage)

dose_photon1 = np.asarray(dose_photon)
dose_neutron1 = np.asarray(dose_neutron)
percentage1 = np.asarray(percentage)

print('\n\n',tb.tabulate(np.column_stack((percentage1,dose_photon1,dose_neutron1)), headers=['Percentage', 'Dose photon', 'Dose neutron'], tablefmt='fancy_outline'))
%timeit tb.tabulate(np.column_stack((percentage1,dose_photon1,dose_neutron1)), headers=['Percentage', 'Dose photon', 'Dose neutron'], tablefmt='fancy_outline')

dose_neutron1 = np.asarray(dose_neutron1).flatten()
print('\n\n',tb.tabulate(np.array([percentage1,dose_photon1.flatten(),dose_neutron1.flatten()]).T, headers=['Percentage', 'Dose photon', 'Dose neutron'], tablefmt='fancy_outline'))

%timeit tb.tabulate(np.array([percentage1,dose_photon1.flatten(),dose_neutron1.flatten()]).T, headers=['Percentage', 'Dose photon', 'Dose neutron'], tablefmt='fancy_outline')


# %% [markdown]
# ---
# ## **Question 13**
# 
# Expliquer ce qu’est l’efficacité biologique relative (EBR en français ou RBE dans la publi
# 
# ---

# %% [markdown]
# **L'efficacité biologique relative (EBR) en radiobiologie**
# 
# L'efficacité biologique relative (EBR) est une mesure qui permet de comparer l'effet biologique de deux rayonnements, à doses absorbées égales. Elle est définie comme le rapport de la dose d'un rayonnement de référence qui produit le même effet que la dose d'un rayonnement considéré.
# 
# Le rayonnement de référence est généralement un rayonnement à faible énergie, tel que les rayons X ou les rayons gamma, pour lequel l'EBR est de 1.
# 
# Les rayonnements à haute énergie, tels que les particules alpha ou les neutrons, ont une EBR supérieure à 1. Cela signifie que, pour une dose absorbée égale, ils produiront un effet biologique plus important.
# 
# 

# %% [markdown]
# ---
# ## **Question 14**
# 
# 14. a- Calculer l’EBR neutrons pour chacune des lignées cellulaires et representez le 
# résultat sous forme d’un même graphique.
# 
# 

# %%
# To calculate the RBE we do the ratio of dose gamma and dose neutron for the same survival fraction

RBE = dose_photon/dose_neutron

for p, rbe in zip(percentage, RBE):
    print(f"RBE for {p}% of survival = {rbe}")

print('\n\n',tb.tabulate(np.array([percentage,np.array(RBE).flatten()]).T, headers=['Percentage', 'RBE'], tablefmt='fancy_outline'))


# %% [markdown]
# b- Vous tracerez l’EBR des neutrons en fonction de la survie de 1 à 95 % par pas de 1%. 
# 
# ---

# %%
percentage = np.linspace(0.01,0.95,95)
RBE =[]
for i in percentage:
    sol_photon = solve_quadratic_equation(param["Photon WSU_DLCL2 exponential"][1],param["Photon WSU_DLCL2 exponential"][0],np.log(i))
    sol_neutron = solve_quadratic_equation(param["Neutron WSU_DLCL2 exponential"][1],param["Neutron WSU_DLCL2 exponential"][0],np.log(i))
    RBE.append(sol_photon/sol_neutron)

RBE = np.asarray(RBE)

plt.figure(figsize=(8,5))

plt.plot(percentage,RBE)
plt.xlabel('Survival fraction')
plt.ylabel('RBE')
plt.title('RBE in function of survival fraction for WSU_DLCL2')
plt.grid(True)
plt.show()


# %% [markdown]
# ---
# ## **Question 15**
# 
# 15. En radiobiologie, on peut réaliser un test simple qui ne nécessite pas la réalisation de courbes de survie cellulaires complètes. Il s’agit du calcul du rapport des survies à 2 Gy. Calculer le rapport des survies à 2 Gy dans notre cas. Commenter la pertinence de cet indicateur radiobiologique dans le cas de la comparaison des irradiations neutron et gamma sur ce type de cellules.
# 
# ---

# %%
photon_2Gy = exponential_model(2,param["Photon WSU_DLCL2 exponential"][0],param["Photon WSU_DLCL2 exponential"][1])
neutron_2Gy = exponential_model(2,param["Neutron WSU_DLCL2 exponential"][0],param["Neutron WSU_DLCL2 exponential"][1])

print(f"Survival fraction for 2 Gy of photon = {photon_2Gy}")
print(f"Survival fraction for 2 Gy of neutron = {neutron_2Gy}")
print(f"Ratio for 2 Gy = {photon_2Gy/neutron_2Gy}",'\n\n')
ratio_2Gy = np.asarray(photon_2Gy/neutron_2Gy)

print(tb.tabulate(np.array([[photon_2Gy],[neutron_2Gy],[ratio_2Gy]]).T, headers=['Survival fraction for 2 Gy of photon','Survival fraction for 2 Gy of neutron', 'Ratio'], tablefmt='fancy_outline'))

# %% [markdown]
# 


