# %% [markdown]
# # Travaux Pratiques de Modélisation pour la Dosimétrie
# 
# ## TP #2 : Étude Radiologique d'un Fantôme Anthropomorphique
# 
# ### Contacts:
# 
# - Véronica Sorgato: [veronica.sorgato88@googlemail.com](mailto:veronica.sorgato88@googlemail.com)
# - Samy Kefs: [samy.kefs@inserm.fr](mailto:samy.kefs@inserm.fr)
# - Yannick Arnoud: [yannick.arnoud@lpsc.in2p3.fr](mailto:yannick.arnoud@lpsc.in2p3.fr)
# 
# ### Données:
# 
# À partir du site du NIST, récupérer dans un fichier Excel les coefficients d'atténuation $ \mu_{att}$ et d'absorption en énergie $\mu_{en}$ des tissus mous, de l'os, du poumon et de l'eau, en fonction de l'énergie des photons, listés dans les fichiers suivants :
# 
# - SoftTissueNIST.xlsx
# - BreastTissueNIST.xlsx
# - CorticalBone-NIST.xlsx
# - LungNIST.xlsx
# - WaterNIST.xlsx
# 
# Ces données sont extraites de la base de données du NIST. Vous disposez également du fichier airNISTxmudat.xlsx qui contient les coefficients d'atténuation, de transmission et d'absorption en énergie pour l'air.
# 
# Densité des tissus:
# - Tissus mous: $0.95 g/cm^{-3}$
# - Tissue mammaires : $1.02 g/cm^{-3}$
# - Os: $1.85 g/cm^{-3}$
# - Poumon: $1.05 g/cm^{-3}$
# - Air: $1.21 \times 10^{-3} g/cm^{-3}$
# 
# On considère des faisceaux de photons parallèles aux énergies suivantes:
# - 20 keV
# - 140 keV
# - 6 MeV
# - 18 MeV
# 
# On considère la géométrie du patient suivante (vue sagittale). Il s'agit d'une vue 2D, et on considérera des pixels de 1x1 mm².
# 
# 
# ![Geométrie](Geometry.jpg)
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tabulate as tb
import seaborn as sns
import time

# %% [markdown]
# ## **Question 1**
# 
# 1. Ouvrir les fichiers excel et tracer en échelle log/log les tissus biologiques (air, os dense, tissu sein, tissu mou et poumon).

# %%
# Path to the Excel file
xlsxFilePath = '/Users/samy/Desktop/PhD/TP_Dosimetrie_M2/TP03/Data_NIST.xlsx'

# %%
# Open the Excel file
xlsxFile = pd.ExcelFile(xlsxFilePath)


# %%
MyDict = {}  # Initialize a dictionary to store data

# Iterate through sheets in the Excel file
for sheet in xlsxFile.sheet_names:
    df = xlsxFile.parse(sheet)  # Parse the current sheet
    if sheet == 'Air':
        # If the sheet is 'Air', store specific columns in the dictionary
        MyDict[sheet] = df[['Energy', 'mu/rho', 'muen/rho', 'mutr/rho']].values.T
    else:
        # For other sheets, store different columns in the dictionary
        MyDict[sheet] = df[['Energy', 'mu/rho', 'muen/rho']].values.T

# Define densities for different materials
BreastDensity = 0.95
AirDensity = 1.21e-3
LungDensity = 1.05
BonesDensity = 1.64
WaterDensity = 1.0
SoftTissueDensity = 1.02

# %%
# Create a figure with 1 row and 2 columns to display the plots side by side
plt.figure(figsize=(15, 7))

# First subplot (on the left) - Plot muen/rho for different materials
plt.subplot(1, 2, 1)
for key, value in MyDict.items():
    plt.plot(value[0], value[2], label='muen/rho for ' + key, linestyle='--')
plt.title('muen/rho')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Energy (MeV)')  # Label for the x-axis
plt.ylabel('mu/rho (cm2/g)')  # Label for the y-axis
plt.legend()  # Display legends
plt.grid()  # Display a grid on the plot

# Second subplot (on the right) - Plot mu/rho for different materials
plt.subplot(1, 2, 2)
for key, value in MyDict.items():
    plt.plot(value[0], value[1], label='mu/rho for ' + key, linestyle='--')
plt.title('mu/rho')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Energy (MeV)')  # Label for the x-axis
plt.ylabel('mu/rho (cm2/g)')  # Label for the y-axis
plt.legend()  # Display legends
plt.grid()  # Display a grid on the plot

# Show the created plots
plt.show()

# %% [markdown]
# ## **Question 2.1**
# 
# 2. Interpoler les coeffiscients des matériaux.
# 
# #### Rappel :
# 
# Pendant le TP2, nous avions constaté deux points importants. Tout d'abord, il était nécessaire d'effectuer l'interpolation en passant à l'échelle logarithmique. Ensuite, nous avions observé que les interpolations en utilisant le logarithme en base 10 et le logarithme naturel donnaient des résultats identiques. Par conséquent, j'ai décidé d'opter pour l'interpolation en utilisant l'échelle logarithmique naturelle.

# %%
EnergyValues = np.array([0.02, 0.14, 6, 18])  # Array of energy values in MeV

# %%
def interpolationValues(energy: np.ndarray, MyDict: dict) -> dict:
    """Interpolation of Values
    Interpolate the values in the MyDict dictionary at the specified energy levels.

    Parameters
    ----------
    energy : np.ndarray
        Energy values in MeV.
    MyDict : dict
        Dictionary containing mu/rho, muen/rho, and mutr/rho values for different materials.

    Returns
    -------
    dict
        Dictionary with interpolated values of mu/rho, muen/rho, and mutr/rho for each material.
    """
    DictInterpolated = {}  # Initialize a dictionary for interpolated values
    for key, value in MyDict.items():
        # Interpolate mu/rho, muen/rho, and mutr/rho at the specified energies
        tmpmu = np.exp(np.interp(np.log(energy), np.log(value[0]), np.log(value[1])))
        tmpmuen = np.exp(np.interp(np.log(energy), np.log(value[0]), np.log(value[2])))
        if key == 'Air':
            # For 'Air', also interpolate mutr/rho
            tmpmutr = np.exp(np.interp(np.log(energy), np.log(value[0]), np.log(value[3])))
            DictInterpolated[key] = np.asarray([tmpmu, tmpmuen, tmpmutr])
        else:
            # For other materials, store only mu/rho and muen/rho
            DictInterpolated[key] = np.asarray([tmpmu, tmpmuen])
          
    return DictInterpolated

DictInterpolated = interpolationValues(EnergyValues, MyDict)  # Interpolate values at specified energy levels


# %%
# Create a figure with 1 row and 2 columns to display the plots side by side
plt.figure(figsize=(15, 7))

# First subplot (on the left) - Plot muen/rho and interpolated values
plt.subplot(1, 2, 1)
for key, value in MyDict.items():
    plt.plot(value[0], value[2], label='muen/rho for ' + key, linestyle='--')
    # Add interpolated values as scatter points
    plt.scatter(EnergyValues, DictInterpolated[key][1], label='muen/rho for ' + key)
plt.title('muen/rho')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Energy (MeV)')  # Label for the x-axis
plt.ylabel('mu/rho (cm2/g)')  # Label for the y-axis
plt.legend()  # Display legends
plt.grid()  # Display a grid on the plot

# Second subplot (on the right) - Plot mu/rho and interpolated values
plt.subplot(1, 2, 2)
for key, value in MyDict.items():
    plt.plot(value[0], value[1], label='mu/rho for ' + key, linestyle='--')
    # Add interpolated values as scatter points
    plt.scatter(EnergyValues, DictInterpolated[key][0], label='mu/rho for ' + key)
plt.title('mu/rho')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Energy (MeV)')  # Label for the x-axis
plt.ylabel('mu/rho (cm2/g)')  # Label for the y-axis
plt.legend()  # Display legends
plt.grid()  # Display a grid on the plot


# %% [markdown]
# ## **Question 2.2**
# 
# 2. En déduire la fluence à l’entrée du fantôme aux 4 énergies
# 
# #### **Rappel**
# 
# \begin{equation}
# Kerma = \phi \times Energy \times \frac{\mu_{tr}}{\rho}
# \end{equation}

# %%
def fluenceComputation(Kerma: float, Dict: dict, energy: np.ndarray) -> np.ndarray:
    """Fluence Computation
    Compute the fluence for a given Kerma value, a dictionary of materials, and an array of energy values.

    Parameters
    ----------
    Kerma : float
        Air Kerma value in Gy (Gray).
    MyDict : dict
        Dictionary of materials with the following keys: 'Air', 'Breast', 'Bones', 'Water', 'SoftTissue' and values: [Energy, mu/rho, muen/rho, mutr/rho].
    energy : np.ndarray
        Array of energy values in MeV (Mega-electronvolts).

    Returns
    -------
    np.ndarray
        Array of fluence values in cm^-2 (square centimeters per unit energy).

    The function calculates fluence using the provided Kerma, energy, and material properties.
    """
    
    # Calculate fluence based on the provided formula
    return Kerma / (energy * 1e6 * 1.6e-19 * Dict['Air'][2] * 1e3)



# %%
Kerma = 5e-3  # Kerma value in Gy (Gray)

flux = fluenceComputation(Kerma, DictInterpolated, EnergyValues)  # Calculate fluence
print(tb.tabulate(np.array([EnergyValues, flux]).T, headers=['Energy (MeV)', 'Flux (cm^-2)'], tablefmt='orgtbl'))


# %% [markdown]
# ## **Question 3**
# 
# 3. Calculer la dose absorbée dans la peau du patient, à l’équilibre électronique à l’entrée du patient pour les 4 faisceaux d’énergies différentes. Les donner dans le compte rendu ainsi que les équations permettant d’y arriver.
# 
# 
# #### **Rappel** :
# 
# \begin{equation}
#     D_{surface}= Energie \times \frac{\mu_{en}}{\rho} \times \phi
# \end{equation}

# %%
# Define a function to compute dose
def compute_dose(mu_en : np.ndarray , rho: float, flux : float , mass : float , energy: np.ndarray) -> np.ndarray:
    """
    Compute the dose at the surface for a given linear attenuation coefficient (mu_en_pred) and a given energy (newEnergy),
    
    Parameters
    ----------
    mu_en : np.ndarray
        The linear attenuation coefficient cm^2/g
    rho : float
        The density g/cm^3
    flux : float
        The flux value cm^-2
    mass : flaot
        The mass value g
    energy : np.ndarray
        The energy values MeV
    Returns 
    -------
    np.ndarray
        The dose values
    """
    return (energy * 1e6 * 1.602e-19 * mu_en * flux) / mass

# %%
SurfaceDose = compute_dose(DictInterpolated['Breast'][1],BreastDensity,flux,1e-3,EnergyValues)

print(tb.tabulate(np.array([EnergyValues,SurfaceDose]).T, headers=['Energy (MeV)', 'Surface Dose (Gy)'], tablefmt='orgtbl'))

# %% [markdown]
# ## **Question 4**
# 
# 4. Construire les quatre fantômes numériques (matrices 2D) contenant les (un fantôme par énergie). Les afficher

# %% [markdown]
# ## **Rappel :**
# 
# L'équation générale d'un cercle dans un plan cartésien est la suivante :
# 
# \begin{equation}
# (x - h)^2 + (y - k)^2 = r^2
# \end{equation}
# 
# où :
# - $(h, k)$ représentent les coordonnées du centre du cercle.
# - $r$ représente le rayon du cercle.
# 
# Cette équation exprime que pour chaque point $(x, y)$ sur le cercle, la somme des carrés des différences entre ses coordonnées et les coordonnées du centre $(h, k)$ est égale au carré du rayon $r^2$.
# 
# Lorsque le centre du cercle est situé à l'origine du système de coordonnées $(h = 0, k = 0)$, l'équation du cercle se simplifie à :
# 
# \begin{equation}
# x^2 + y^2 = r^2
# \end{equation}
# 
# Cela correspond à un cercle centré à l'origine avec un rayon $r$.
# ___

# %% [markdown]
# Dans le code suivant, je teste diverses méthodes pour simuler un demi-cercle, sachant qu'il en existe de nombreuses autres. L'objectif est de déterminer la méthode la plus rapide.

# %%
# Method c() - Using nested loops
def c():
    # Create a 2D NumPy array of zeros with dimensions (450, 1800)
    a = np.zeros((450, 1800))
    centrex = 625  # Define the x-coordinate of the center of the circle
    for i in range(450):
        for j in range(1800):
            # Check if the distance from the current point to the center is within the circle's radius
            if (i - centrex)**2 + (a.shape[1] - j)**2 <= 425**2:
                a[i, j] = 1  # Set the value to 1 if inside the circle
    return a

# Method d() - Using meshgrid and a mask
def d():
    # Create a 2D NumPy array of zeros with dimensions (1800, 450)
    a = np.zeros((1800, 450))
    centrex = 625  # Define the x-coordinate of the center of the circle
    # Generate grid coordinates using meshgrid
    x, y = np.meshgrid(np.arange(450), np.arange(1800))
    # Create a mask to identify points within the circle
    mask = (450 - x)**2 + (y - centrex)**2 <= 425**2
    a[mask] = 1  # Set the identified points to 1
    return a

# Method e() - Using vectorized operations and broadcasting
def e():
    # Create a 2D NumPy array of zeros with dimensions (1800, 450)
    a = np.zeros((1800, 450))
    centrey = 625  # Define the y-coordinate of the center of the circle
    rayon = 425  # Define the radius of the circle
    # Generate 1D arrays for x and y coordinates
    x = np.arange(450)
    y = np.arange(1800)[:, np.newaxis]  # Use broadcasting to create a 2D array for y
    # Create a condition to identify points within the circle
    condition = (450 - x)**2 + (y - centrey)**2 <= rayon**2
    # Use np.where to set points inside the circle to 1 based on the condition
    a = np.where(condition, 1, a)
    return a

# Measure the execution time of each method using %timeit
%timeit c()
%timeit d()
%timeit e()


# %% [markdown]
# **NOTE**
# On utilisera donc la 3eme méthode.

# %%
# Define a function to create a half circle and add it to a given array
def HalfCircleFunc(Part1, Center, Radius):
    x = np.arange(Part1.shape[1])  # Generate x-coordinates
    y = np.arange(Part1.shape[0])[:, np.newaxis]  # Generate y-coordinates

    # Define a condition to identify points within the half circle
    condition = (Part1.shape[1] - x)**2 + (y - Center)**2 <= Radius**2

    # Set points inside the half circle to 3 (a custom value)
    return np.where(condition, 3, Part1)

# Initialize arrays for different parts of the image
Part1 = np.zeros((1800, 450))  # Part 1 (initially all zeros)
Part2 = np.zeros((1800, 150))  # Part 2 (initially all zeros)
Part3 = np.zeros((1800, 150))  # Part 3 (initially all zeros)
Part4 = np.ones((1800, 250))  # Part 4 (initially all ones for Lung)
Part5 = np.ones((1800, 800))  # Part 5 (initially all ones)
Part6 = np.ones((1800, 650))  # Part 6 (initially all ones for Lung)
Part7 = np.ones((1800, 100)) * 4  # Part 7 (initially all fours for Bone)
Part8 = np.ones((1800, 150)) * 2  # Part 8 (initially all twos for Soft)

#Modify values in Part1 based on specific regions
Part1 = HalfCircleFunc(Part1,650,450)

# Modify values in Part2, Part3, and Part5 based on specific regions
Part2[0:200, ...] = 2  # Set to 2 (Soft)
Part2[200:1100, ...] = 3  # Set to 3 (Breast)
Part2[1100:, ...] = 2  # Set to 2 (Soft)

Part3[0:300, ...] = 2  # Set to 2 (Soft)
Part3[300:500, ...] = 4  # Set to 4 (Bone)
Part3[500:800, ...] = 2  # Set to 2 (Soft)
Part3[800:1000, ...] = 4  # Set to 4 (Bone)
Part3[1000:1300, ...] = 2  # Set to 2 (Soft)
Part3[1300:1500, ...] = 4  # Set to 4 (Bone)
Part3[1500:, ...] = 2  # Set to 2 (Soft)

Part5[800:1600, ...] = 2  # Set to 2 (Soft)

# Concatenate the parts to create the final image
FinalOne = np.concatenate((Part1, Part2, Part3, Part4, Part5, Part6, Part7, Part3, Part8), axis=1)






# Create a legend for the values in the image
legends = {
    0: 'Air',
    1: 'Lung',
    2: 'Soft',
    3: 'Breast',
    4: 'Bone',
}


# Display the image with the legend
plt.imshow(FinalOne, cmap= 'Pastel1' , vmin=0, vmax=4)
cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4])
cbar.set_ticklabels([legends[0], legends[1], legends[2], legends[3], legends[4]])
plt.show()





# %% [markdown]
# ## **Question 5 et 6**
# 
# 5. Construire les quatre fantômes numériques (matrices 2D) contenant les (un fantôme par énergie). Les afficher.
# 6. Construire le fantôme numérique (matrice 2D) contenant les masses volumiques. L’afficher.

# %% [markdown]
# ___
# Ici, je présente une méthode simple de création d'un dictionnaire contenant les différents coefficients.

# %%
phantom_muAtt = {}
phantom_muEn = {}

for index in range(len(EnergyValues)):
    # Mu Attenuation Coefficient
    phant_tmp = np.zeros_like(FinalOne)
    phant_tmp = np.where(FinalOne == 0, 0, 0) # Air
    phant_tmp = np.where(FinalOne == 1, DictInterpolated['Lung'][0][index], phant_tmp) # Lung
    phant_tmp = np.where(FinalOne == 2, DictInterpolated['Soft'][0][index], phant_tmp) # Soft
    phant_tmp = np.where(FinalOne == 3, DictInterpolated['Breast'][0][index], phant_tmp) # Breast
    phant_tmp = np.where(FinalOne == 4, DictInterpolated['Bones'][0][index], phant_tmp) # Bone
    phantom_muAtt[EnergyValues[index]] = phant_tmp

    # Mu En Absorption Coefficient
    
    phant_tmp = np.zeros_like(FinalOne)
    phant_tmp = np.where(FinalOne == 0, 0, 0) # Air
    phant_tmp = np.where(FinalOne == 1, DictInterpolated['Lung'][1][index], phant_tmp) # Lung
    phant_tmp = np.where(FinalOne == 2, DictInterpolated['Soft'][1][index], phant_tmp) # Soft
    phant_tmp = np.where(FinalOne == 3, DictInterpolated['Breast'][1][index], phant_tmp) # Breast
    phant_tmp = np.where(FinalOne == 4, DictInterpolated['Bones'][1][index], phant_tmp) # Bone
    phantom_muEn[EnergyValues[index]] = phant_tmp

    
# Density
phantom_density = np.zeros_like(FinalOne)
phantom_density = np.where(FinalOne == 0, 0, 0) # Air
phantom_density = np.where(FinalOne == 1, LungDensity, phantom_density) # Lung
phantom_density = np.where(FinalOne == 2, SoftTissueDensity, phantom_density) # Soft
phantom_density = np.where(FinalOne == 3, BreastDensity, phantom_density) # Breast
phantom_density = np.where(FinalOne == 4, BonesDensity, phantom_density) # Bone


    
    

# %% [markdown]
# ___
# Il existe certainement d'autres méthodes plus performantes et plus concises pour accomplir la même tâche.
# Il convient de noter que les tracés graphiques font partie des fonctions qui peuvent consommer beaucoup de mémoire et de temps. De plus, leur utilisation avec Seaborn peut être plus gourmande en temps, en particulier avec de grandes matrices.

# %%
# Initialize dictionaries to store attenuation coefficients, 
# energy absorption coefficients, and densities
phantom_muAtt1 = {}
phantom_muEn1 = {}

plt.figure(figsize=(20, 15))
for index in range(len(EnergyValues)):
    # Mu Attenuation Coefficient
    phant_tmp = np.zeros_like(FinalOne)
    phant_tmp = np.select([FinalOne == 0, FinalOne == 1, FinalOne == 2, FinalOne == 3, FinalOne == 4],
                           [0, DictInterpolated['Lung'][0][index], DictInterpolated['Soft'][0][index],
                            DictInterpolated['Breast'][0][index], DictInterpolated['Bones'][0][index]])
    
    phantom_muAtt1[EnergyValues[index]] = phant_tmp
    plt.subplot(2, len(EnergyValues), index + 1)
    sns.heatmap(phantom_muAtt1[EnergyValues[index]], cmap='Spectral',xticklabels=False, yticklabels=False)
    plt.title('Mu Attenuation Coefficient at ' + str(EnergyValues[index]) + ' MeV')
    
    
    # Mu En Absorption Coefficient
    phant_tmp = np.zeros_like(FinalOne)
    phant_tmp = np.select([FinalOne == 0, FinalOne == 1, FinalOne == 2, FinalOne == 3, FinalOne == 4],
                           [0, DictInterpolated['Lung'][1][index], DictInterpolated['Soft'][1][index],
                            DictInterpolated['Breast'][1][index], DictInterpolated['Bones'][1][index]])

    phantom_muEn1[EnergyValues[index]] = phant_tmp
    plt.subplot(2, len(EnergyValues), index + 5)
    sns.heatmap(phantom_muEn1[EnergyValues[index]], cmap='Spectral',xticklabels=False, yticklabels=False)
    plt.title('Mu En Absorption Coefficient at ' + str(EnergyValues[index]) + ' MeV')
plt.show()


    # Density
phantom_density1 = np.zeros_like(FinalOne)
phantom_density1 = np.select([FinalOne == 0, FinalOne == 1, FinalOne == 2, FinalOne == 3, FinalOne == 4],
                           [0, LungDensity, SoftTissueDensity, BreastDensity, BonesDensity])



# %% [markdown]
# ### **Informations**
# 
# Dans les sections à venir, nous allons explorer l'utilisation pratique et optimale de deux types de matrices : la matrice de distance et la matrice de fluence. Ces matrices stockeront respectivement la distance à l'intérieur du matériau en excluant l'air, ainsi que la fluence à chaque point de la matrice.

# %%
# Create a distance map , where each pixel value is the distance to the nearest non-zero pixel
DistanceArray = np.zeros_like(FinalOne)
for i in range (DistanceArray.shape[0]):
    row = phantom_density[i, :]
    
    non_zero_indices = np.where(row != 0)[0]
    
    DistanceArray[i, non_zero_indices] = np.arange(1, len(non_zero_indices) + 1)
plt.imshow(DistanceArray)
plt.colorbar()
plt.show()

# %% [markdown]
# ## Trés important (à lire)
# Ceci est un extrait du livre **"Python Data Science Handbook"** de Jake VanderPlas! [Livre](https://jakevdp.github.io/PythonDataScienceHandbook/) et [JupyterNote Book](https://github.com/jakevdp/PythonDataScienceHandbook) disponible gratuitement.
# 
# **Je ne peux que vous conseiller de lire cette ouvrage.**
# 
# > Computation on NumPy arrays can be very fast, or it can be very slow. The key to making it fast is to use vectorized operations, generally implemented through NumPy's universal functions (ufuncs). This section motivates the need for NumPy's ufuncs, which can be used to make repeated calculations on array elements much more efficient. It then introduces many of the most common and useful arithmetic ufuncs available in the NumPy package.
# 
# > The Slowness of Loops
# > Python's default implementation (known as CPython) does some operations very slowly. This is in part due to the dynamic, interpreted nature of the language: the fact that types are flexible, so that sequences of operations cannot be compiled down to efficient machine code as in languages like C and Fortran. Recently there have been various attempts to address this weakness: well-known examples are the PyPy project, a just-in-time compiled implementation of Python; the Cython project, which converts Python code to compilable C code; and the Numba project, which converts snippets of Python code to fast LLVM bytecode. Each of these has its strengths and weaknesses, but it is safe to say that none of the three approaches has yet surpassed the reach and popularity of the standard CPython engine.

# %% [markdown]
# ## **Rappel**
# 
# 
# \begin{equation}
#     \phi_{depth} =  \phi_{depth-1} \times e^{\mu_{att} \times depth}
# \end{equation}
#  

# %%
#  Create a Fluence map , where each pixel value is the fluence at that pixel using for loop 
def MatriceOfFLux(phantom_muAtt:dict, phantom_density:np.ndarray, flux_entre:np.ndarray, info: np.ndarray, energy:np.ndarray)-> np.ndarray:
    """
    Compute the fluence at each pixel of the phantom using the provided muAttenuation, density, flux, info, and energy values.
    
    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary containing muAttenuation values for different energies.
    phantom_density : np.ndarray
        Array containing density values for each pixel of the phantom.
    flux_entre : np.ndarray
        Array containing flux values for different energies.
    info : np.ndarray
        Array containing information about the phantom.
    energy : np.ndarray
        Array containing energy values.
    
    Returns
    -------
    np.ndarray
        Array containing fluence values for different energies.
    """
    
    # Create an empty matrix to store fluence
    MatriceOfFlux = np.zeros((len(flux_entre), phantom_muAtt[18].shape[0], phantom_muAtt[18].shape[1]))
    
    # Iterate over each value in flux_entre
    for index, flux_entre in enumerate(flux_entre):
        # Iterate over row indices (i) of the phantom_muAtt matrix
        for i in range(phantom_muAtt[18].shape[0]):
            # Initialize the variable 'flux' with the current value of flux_entre
            flux = flux_entre
            # Iterate over column indices (j) of the phantom_muAtt matrix
            for j in range(phantom_muAtt[18].shape[1]):
                # Check if the value of info at position (i, j) is not equal to zero
                if info[i, j] != 0:
                    # Update the value of MatriceOfFlux at position (index, i, j-1) with the current value of 'flux'
                    MatriceOfFlux[index, i, j-1] = flux
                    # Update the value of 'flux' by multiplying it by exp(-phantom_muAtt * phantom_density * 1e-2)
                    flux = flux * np.exp(-phantom_muAtt[energy[index]][i, j] * phantom_density[i, j] * 1e-2)

    return MatriceOfFlux

        

# %%
time_start = time.time()
MatrixFlux = MatriceOfFLux(phantom_muAtt, phantom_density, flux, DistanceArray,EnergyValues)
time_end = time.time()
print('Time taken for loop method: ', time_end - time_start, 'seconds')

# %%
plt.figure(figsize=(20, 15))
for index in range(len(EnergyValues)):
    plt.subplot(2, 2, index + 1)
    plt.imshow(MatrixFlux[index])
    plt.title('Matrix of flux at ' + str(EnergyValues[index]) + ' MeV')
    plt.colorbar()
plt.show()

# %% [markdown]
# ___
# Comme expliqué précédemment, les boucles `for` sont très chronophages, il est donc nécessaire d'essayer de les éviter autant que possible. Cela permettra d'économiser du temps. Je vous propose ici une alternative beaucoup moins coûteuse en temps. Bien sûr, les boucles `for` ne sont pas complètement éliminées, mais cette approche alternative est simple et très performante.

# %%
def MatriceOfFLuxV2(phantom_muAtt: dict, phantom_density: np.ndarray, flux_entre: np.ndarray, info: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Compute the fluence at each pixel of the phantom using the provided muAttenuation, density, flux, info, and energy values.
    
    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary containing muAttenuation values for different energies.
    phantom_density : np.ndarray
        Array containing density values for each pixel of the phantom.
    flux_entre : np.ndarray
        Array containing flux values for different energies.
    info : np.ndarray
        Array containing information about the phantom.
    energy : np.ndarray
        Array containing energy values.
    
    Returns
    -------
    np.ndarray
        Array containing fluence values for different energies.
    """
    
    # Create a 3D array to store the fluence values
    MatriceOfFLux = np.ones((len(flux_entre), phantom_muAtt[18].shape[0], phantom_muAtt[18].shape[1]))
    
    # Iterate over each flux value
    for index, flux_entre in enumerate(flux_entre):
        flux = flux_entre

        # Initialize the fluence values with the current flux value
        MatriceOfFLux[index, :, :] *= flux_entre
        
        # Calculate the product of muAttenuation and density for the current energy
        muATT = phantom_muAtt[energy[index]] * phantom_density
        
        # Iterate over the columns of the muAttenuation matrix
        for j in range(phantom_muAtt[18].shape[1]):
            if j != 0:
                # Update the fluence values using exponential attenuation
                MatriceOfFLux[index, :, j] = MatriceOfFLux[index, :, j-1] * np.exp(-muATT[:, j] * 1e-2)
        
        # Multiply the fluence values by the information array
        MatriceOfFLux[index, :, :] = MatriceOfFLux[index, :, :] * info

    return MatriceOfFLux


# %%
time_start = time.time()
test = np.where(DistanceArray != 0, 1, 0)
MatriceOfFlux2 = MatriceOfFLuxV2(phantom_muAtt, phantom_density, flux, test ,EnergyValues)
time_end = time.time()
print('Time taken for vectorized method: ', time_end - time_start, 'seconds')

# %%
plt.figure(figsize=(20, 15))
for i in range(MatriceOfFlux2.shape[0]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(MatriceOfFlux2[i,:,:])
    plt.title('Matrix of flux at ' + str(EnergyValues[i]) + ' MeV')
    plt.colorbar()
plt.show()

# %% [markdown]
# ## **Question 7**
# 
# 7. Calculer et tracer la variation de la dose due au rayonnement primaire en profondeur selon l’axe en pointillé.
# 
# \begin{equation}
#     D_{depth parallel}= Energie \times \frac{\mu_{en}}{\rho} \times \phi_{depth}
# \end{equation}

# %%
def Compute_dose_depth_axe(phantom_muAtt:dict,phantom_muEn:dict,phantom_density:np.ndarray,flux:np.ndarray,energy:np.ndarray,dist:np.ndarray)->np.ndarray:
    """Compute_dose_depth_axe
    Compute the dose depth curve for a given phantom

    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary of mu attenuation coefficient for each material
    phantom_muEn : dict
        Dictionary of mu energy absorption coefficient for each material
    phantom_density : dict
        Dictionary of density for each material
    flux : np.ndarray
        Array of flux values
    energy : np.ndarray
        Array of energy values
    dose_surface : np.ndarray
        Array of dose surface values
    dist : np.ndarray
        Array of distance values

    Returns
    -------
    np.ndarray
        Array of dose depth values
    """
    res = []
    
    for index,energy_val in enumerate(energy):

        res.append(energy_val*1e6*1.6e-19*phantom_muEn[energy_val][900,:] *1e3* flux[index,900,:]) 
            
        
    return np.asarray(res)

# %%
Dose_axe = Compute_dose_depth_axe(phantom_muAtt,phantom_muEn,phantom_density,MatriceOfFlux2,EnergyValues,DistanceArray)


for i in range (len(EnergyValues)):
    
    plt.plot((Dose_axe[i,:]/np.max(Dose_axe[i,:])),label = str(EnergyValues[i])+' MeV')
plt.legend()
plt.title('Dose depth curve')
plt.xlabel('Depth')
plt.ylabel('Dose percentage')
plt.show()


plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.plot((Dose_axe[i,:]),label = str(EnergyValues[i])+' MeV')
    
    plt.yscale('log')
    plt.legend()
    plt.title('Dose depth curve')
    plt.xlabel('Depth')
    plt.ylabel('Dose (Gy)')
plt.show()

# %% [markdown]
# ## **Question 8**
# 
# 8. Calculer les cartes de dose primaire aux quatre énergies.
# 
# \begin{equation}
#     D_{depth parallel}= Energie \times \frac{\mu_{en}}{\rho} \times \phi_{depth}
# \end{equation}

# %% [markdown]
# **NOTE** : Le code actuel utilise trois boucles `for`. Nous allons explorer des moyens de le rendre plus efficace. Actuellement, nous manipulons des matrices de petite taille, mais si l'on considère une matrice 10 fois plus grande, le temps d'exécution augmentera presque de façon proportionnelle, soit près de 10 fois plus long.

# %%
def DoseMap(phantom_muAtt: dict, phantom_muEn: dict, phantom_density: np.ndarray, flux: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Compute the dose map for a given phantom.

    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary of mu attenuation coefficient for each material.
    phantom_muEn : dict
        Dictionary of mu energy absorption coefficient for each material.
    phantom_density : np.ndarray
        Array of density values for the phantom.
    flux : np.ndarray
        Array of flux values for different energies.
    energy : np.ndarray
        Array of energy values.
    
    Returns
    -------
    np.ndarray
        Array of dose depth values.
    """
    dose_map = np.zeros((len(energy), phantom_density.shape[0], phantom_density.shape[1]))
    
    # Iterate over each energy level
    for index, energy_val in enumerate(energy):
        
        # Iterate over rows of mu energy absorption coefficient
        for i in range(phantom_muEn[energy_val].shape[0]):
            dist = 0
            flux1 = flux[index]
            
            # Iterate over columns of mu energy absorption coefficient
            for j in range(phantom_muEn[energy_val].shape[1]):
                if phantom_muEn[energy_val][i, j] == 0:
                    dist = 0
                else:
                    dist += 1
                
                # Calculate dose depth value
                dose_map[index, i, j] = (energy[index] * 1e6 * 1.6e-19 * phantom_muEn[energy_val][i, j] * 1e3 * flux1)
                flux1 = flux1 * np.exp(-phantom_muAtt[energy_val][i, j] * phantom_density[i, j] * 1e-2)
                    
    return dose_map


# %%
time_start = time.time()
DoseMapp = DoseMap(phantom_muAtt,phantom_muEn,phantom_density,flux,EnergyValues)
time_end = time.time()
print('Time taken for loop method: ', time_end - time_start, 'seconds')

# %% [markdown]
# ___
# Pour le plot, j'ajoute les isodoses, vous pouvez essayer de les modifier, et ajouter la legende correpondante.

# %%
isodose_levels = [90, 80, 50, 20]

c = ['blue', 'green', 'white', 'red']

plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    
    plt.imshow(DoseMapp[i,:,:],cmap='jet')
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
    plt.colorbar()
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
    for level, color in zip(isodose_levels,c):
        
        dose_level = level / 100 * np.max(DoseMapp[i, :, :])
        plt.contour(DoseMapp[i, :, :], levels=[dose_level],colors=color, linestyles='solid', linewidths=2)
        
plt.show()

# %% [markdown]
# ## **Informations**
# 
# Pour les mêmes raisons évoquées précédemment, notre objectif est d'optimiser le code en réduisant l'utilisation de boucles 'for' et en utilisant des UFuncs ou des opérateurs vectorisés

# %%
def DoseMapv2(phantom_muAtt: dict, phantom_muEn: dict, phantom_density: np.ndarray, flux: np.ndarray, energy: np.ndarray,dist:np.ndarray) -> np.ndarray:
    """DoseMap
    Compute the dose map for a given phantom

    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary of mu attenuation coefficient for each material
    phantom_muEn : dict
        Dictionary of mu energy absorption coefficient for each material
    phantom_density : dict
        Dictionary of density for each material
    flux : np.ndarray
        Array of flux values
    energy : np.ndarray
        Array of energy values
    dose_surface : np.ndarray
        Array of dose surface values
    dist : np.ndarray
        Array of distance values

    Returns
    -------
    np.ndarray
        Array of dose depth values
    """
    dose_map = np.zeros((len(energy), phantom_density.shape[0], phantom_density.shape[1]))
    
    for index, energy_val in enumerate(energy):
        
        
        dose_map[index, :, :] = (
            energy_val * 1e6 * 1.6e-19 * phantom_muEn[energy_val] * 1e3 *
            flux[index, :, :] 
        )
    
    return np.asarray(dose_map)

# %%
time_start = time.time()
DoseMapp = DoseMapv2(phantom_muAtt,phantom_muEn,phantom_density,MatrixFlux,EnergyValues,DistanceArray)
time_end = time.time()
print('Time taken for vectorized method: ', time_end - time_start, 'seconds')

# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(DoseMapp[i,:,:],cmap='jet')
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
    plt.colorbar()
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
plt.show()

# %% [markdown]
# 
# ## **Utilisation de Numba et des decorateurs :**
# 
# - **Numba (`@jit`)** : Numba est un compilateur JIT (Just-In-Time) pour Python. L'utilisation du décorateur `@jit` indique à Numba de compiler une fonction Python en code machine natif. Voici quelques points importants à noter :
# 
#   - Le compilateur Numba analyse le code de la fonction Python et génère du code machine optimisé pour la plateforme d'exécution. Cela permet d'accélérer considérablement l'exécution de la fonction.
# 
#   - Le décorateur `@jit` peut être utilisé sans arguments, mais il offre également des options telles que `nopython=True` et `parallel=True` pour contrôler le comportement de la compilation.
# 
# - **`nopython=True`** : Lorsque vous utilisez `@jit(nopython=True)`, Numba tente de forcer la compilation en mode "No Python". Cela signifie que la fonction compilée ne doit pas utiliser de fonctionnalités Python non prises en charge, telles que des listes Python pures ou des boucles Python. Le mode "No Python" permet d'obtenir les meilleures performances car le code Python est évité autant que possible.
# 
# - **`parallel=True`** : Lorsque vous utilisez `@jit(parallel=True)`, Numba tente de paralléliser automatiquement les boucles dans la fonction. Cela signifie que Numba peut répartir les itérations de la boucle sur plusieurs cœurs de processeur si cela est possible, ce qui accélère le calcul.
# 
# - **`prange`** : `prange` est une fonction de Numba conçue pour être utilisée à l'intérieur de fonctions décorées avec `@jit(parallel=True)`. Elle est similaire à la fonction `range` de Python, mais elle est destinée à être utilisée dans des boucles parallèles. L'utilisation de `prange` permet d'exploiter la puissance de traitement parallèle des processeurs multi-cœurs.
# 
# En combinant ces fonctionnalités, Numba offre un moyen efficace d'accélérer le code Python, en compilant des fonctions en code machine, en évitant le code Python pur autant que possible et en parallélisant les boucles pour tirer parti des architectures multi-cœurs. Cela fait de Numba un outil puissant pour les calculs scientifiques et numériques en Python.

# %%
import numba as nb
from numba import prange

@nb.jit(nopython=True, parallel=True)
def DoseMapV3(phantom_muAtt:dict,phantom_muEn:dict,phantom_density:np.ndarray,flux:np.ndarray,energy:np.ndarray,dist:np.ndarray)->np.ndarray:
    """DoseMap
    Compute the dose map for a given phantom

    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary of mu attenuation coefficient for each material
    phantom_muEn : dict
        Dictionary of mu energy absorption coefficient for each material
    phantom_density : dict
        Dictionary of density for each material
    flux : np.ndarray
        Array of flux values
    energy : np.ndarray
        Array of energy values
    dose_surface : np.ndarray
        Array of dose surface values

    Returns
    -------
    np.ndarray
        Array of dose depth values
    """
    dose_map = np.zeros((phantom_density.shape[0],phantom_density.shape[1]))
    
    for i in prange (phantom_muEn.shape[0]):
        for j in prange (phantom_muEn.shape[1]):
            
            dose_map[i,j]= (energy*1e6*1.6e-19*phantom_muEn[i,j] *1e3*
                            flux[i,j])
                
        
    return dose_map

# %%
plt.figure(figsize=(20, 15))
for index, energy_val in enumerate(EnergyValues):
    time_start = time.time()
    flux2 = MatrixFlux[index,:,:]
    phantom_muAtt2 = phantom_muAtt[energy_val]
    phantom_muEn2 = phantom_muEn[energy_val]
    DoseMapp = DoseMapV3(phantom_muAtt2,phantom_muEn2,phantom_density,flux2,energy_val,DistanceArray)
    time_end = time.time()
    print('Time taken for Numba method: ', time_end - time_start, 'seconds')
    plt.subplot(2, 2, index + 1)
    plt.imshow(DoseMapp,cmap='jet')
    plt.title('Dose map for '+str(energy_val)+' MeV')
    plt.colorbar()
      
plt.show()

# %% [markdown]
# ## **Question 9.1**
# 
# 9. On souhaite, pour une application en radiologie, avoir une fluence en sortie de fantôme, de 3000 photons/mm2. 
# 
# 
# #### **Rappel :**
# 
# \begin{equation}
#     \phi_{depth-1} =  \frac{\phi_{depth}}{e^{-\mu_{att} \times depth}}
# \end{equation}
# 
# ### **Les différentes possibilitées :**
# 
# Cette question peut être abordée de plusieurs manières différentes. Tout d'abord, nous souhaitons calculer la fluence le long de l'axe pointillé tracé. Une autre approche consisterait à considérer que nous avons au minimum 3000 photons en sortie du fantôme. La question est de déterminer quelle serait la fluence minimal en entrée pour obtenir ces 3000 photons au minimum en tout poins en tenant compte de l'ensemble du fantôme.
# 
# Pour résoudre ce problème, nous pourrions envisager différentes approches. L'une d'entre elles serait d'utiliser le calcul inverse des matrices de flux, mais cela pourrait être très long et peu pratique. Une autre méthode consisterait à effectuer des calculs ligne par ligne et à rechercher le flux d'entrée maximal.
# 
# 

# %%
# Caluculate the dose map using the function DoseMapV3 which uses fluence matrix
def ComputeFluxV1(phantom_muAtt: dict, phantom_density: np.ndarray, flux_entre: float, info: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Calculate the fluence using given muAttenuation, density, flux, info, and energy values.
    
    Parameters
    ----------
    phantom_muAtt : dict
        A dictionary containing muAttenuation values for different energies.
    phantom_density : np.ndarray
        An array containing density values for each pixel of the phantom.
    flux_entre : np.ndarray
        An array containing the input flux values for different energies.
    info : np.ndarray
        An array containing information about the phantom.
    energy : np.ndarray
        An array containing energy values.
    
    Returns
    -------
    np.ndarray
        An array containing fluence values for different energies.
    
    Description
    -----------
    This function calculates the fluence at each pixel of the phantom. It iterates over different energy values, calculates the fluence, and updates the MatriceOfFlux accordingly. The fluence is computed based on muAttenuation, density, input flux, and information values.
    """
    MatriceOfFlux = np.ones((len(energy), phantom_muAtt[18].shape[0], phantom_muAtt[18].shape[1]))
    info = info[:, ::-1]
    
    # Iterate over different energy values
    for index, energy_val in enumerate(energy):
        
        MatriceOfFlux[index, :, :] *= flux_entre
        muATT = phantom_muAtt[energy_val] * phantom_density
        muATT = muATT[:,::-1]
        MatriceOfFlux[index, :, :] = MatriceOfFlux[index, :, ::-1]
        # Iterate over pixel columns
        for j in range(phantom_muAtt[18].shape[1]):
            if j != 0:
                # Calculate fluence for the current pixel
                MatriceOfFlux[index, :, j] = MatriceOfFlux[index, :, j - 1] / np.exp(-muATT[:, j] * 1e-2)
        
        # Multiply fluence by information values
        MatriceOfFlux[index, :, :] = MatriceOfFlux[index, :, :] * info
        
    MatriceOfFlux = MatriceOfFlux[:, :, ::-1]
                
    return MatriceOfFlux


# %%
flux_end=3e5
time_start = time.time()
info = np.where(DistanceArray != 0, 1, 0)
MatrixFlux = ComputeFluxV1(phantom_muAtt, phantom_density, flux_end, info,EnergyValues)
time_end = time.time()
print('Time taken for loop method: ', time_end - time_start, 'seconds')

# %%
#Just to check if the function is working, we print the flux at the last pixel of the phantom to see if it's equal to 300000ph/cm2
print(MatrixFlux[1,900,-1])

# %%
flux_entre = np.zeros((len(EnergyValues)))
flux_entreaxial = np.zeros((len(EnergyValues)))
for i in range (len(EnergyValues)):
    flux_entre[i] = np.max(MatrixFlux[i,:,:])
    flux_entreaxial[i] = np.max(MatrixFlux[i,900,:])

print(tb.tabulate(np.array([EnergyValues,flux_entre,flux_entreaxial]).T, headers=['Energy (MeV)', 'Flux (cm-2)','Flux (cm-2) Axe'], tablefmt='orgtbl'))


# %%
# Calculate fluence along an axis
def compute_Flux_Axial(phantom_muAtt:dict, phantom_density:dict, flux_end:float,EnergyValues:np.ndarray)-> float:
    """compute_Flux
    Compute the flux at the end of the phantom
    
    Parameters
    ----------
    phantom_muAtt : dict
        Dictionary of mu attenuation coefficient for each material
    phantom_density : dict
        Dictionary of density for each material
    flux_end : float
        The flux value at the end of the phantom
    EnergyValues : np.ndarray
        Array of energy values
        
    Returns
    -------
    float
        The flux value at the end of the phantom
    """
    
    flux_entre = []
    for energy in EnergyValues:
        Axe_muAtt = phantom_muAtt[energy][900,...]*phantom_density[900,...]
        Axe_muAtt = Axe_muAtt[::-1]
        value , counts = np.unique(Axe_muAtt, return_counts=True)
        # print(value,counts)
        
        muAttEq = ((value*counts)/counts.sum()).sum()
        # print(flux_end)
        
        
        flux_entre.append(
            flux_end/
                   (np.exp(-muAttEq * counts.sum() * 1e-2)))
        
    
    return np.asarray(flux_entre)

# %%
flux_end=3e5
time_start = time.time()
flux_entre_axis = compute_Flux_Axial(phantom_muAtt,phantom_density,flux_end,EnergyValues)
time_end = time.time()
print('Time taken for axial method: ', time_end - time_start, 'seconds')
print(tb.tabulate(np.array([EnergyValues,flux_entre_axis]).T, headers=['Energy (MeV)', 'Flux (cm-2)'], tablefmt='orgtbl'))

# %% [markdown]
# ## **Question 9.2**
# 
# 9. Calculez la dose en entrée correspondante. Commentez.
# 

# %%
Dose_surface = compute_dose(DictInterpolated['Breast'][1],BreastDensity,flux_entre,1e-3,EnergyValues)
Dose_surface = np.asarray(Dose_surface)
print(tb.tabulate(np.array([EnergyValues,Dose_surface]).T, headers=['Energy (MeV)', 'Surface Dose (Gy)'], tablefmt='orgtbl'))

# %% [markdown]
# ## **Question 10**
# 
# 10. Pour cette même dose en entrée calculée dans 9., calculer la dose moyenne au cœur et au sein. Commentez.

# %%
# First we calculate the matrix of flux using the function MatriceOfFLuxV2
test = np.where(DistanceArray != 0, 1, 0)
NewMatrixFlux = MatriceOfFLuxV2(phantom_muAtt, phantom_density, flux_entre, test ,EnergyValues)

# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(NewMatrixFlux[i,:,:])
    plt.title('Matrix of flux at ' + str(EnergyValues[i]) + ' MeV')
    plt.colorbar()
plt.show()
  

# %%
# Then, we calculate the dose map using the function DoseMapv2
Dose_Map_New_flux= DoseMapv2(phantom_muAtt,phantom_muEn,phantom_density,NewMatrixFlux,EnergyValues,DistanceArray)

# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Dose_Map_New_flux[i,:,:],cmap='jet')
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
    plt.colorbar()
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
plt.show()

# %%
# Initialize lists to store mean doses for the heart and breast
HeartMean = []
BreastMean = []

# Iterate through different energy values
for i in range(len(EnergyValues)):
    # Calculate the mean dose in the heart region (rows 800 to 1600, columns 1000 to 1800)
    HeartMean.append(np.mean(Dose_Map_New_flux[i, 800:1600, 1000:1800]))
    
    # Find indices where the phantom's density is equal to BreastDensity
    BreastIndices = np.where(phantom_density == BreastDensity)
    
    # Calculate the mean dose in the breast region using the indices found earlier
    BreastMean.append(np.mean(Dose_Map_New_flux[i, BreastIndices[0], BreastIndices[1]]))


print(tb.tabulate(np.array([EnergyValues, HeartMean, BreastMean]).T, headers=['Energy (MeV)', 'Heart Mean (Gy)', 'Breast Mean (Gy)'], tablefmt='orgtbl'))


# %% [markdown]
# ## **Question 11**
# 
# 11. Calculez les coefficients massiques d'atténuation et d'absorption en énergie d’un poumon plus réaliste et plein d'air (fraction massique de poumon: 78% ; fraction massique d'air 22%) en fonction de l'énergie. Masse volumique 0,35 g/cm3. Refaire les mêmes simulations avec ce poumon plein d’aire, afin d'obtenir les rendements, cartes de doses et profils d'intensité pour les quatre énergies demandées aux questions 6 à 10.

# %%
# Calculate the new numerical phantoms with the new values for lung

for index, energy in enumerate(EnergyValues):
    
    print('muAtt')
    print('Energy : ',energy)
    print('Lung : ',DictInterpolated['Lung'][0][index])
    print('Air : ',DictInterpolated['Air'][0][index])
    print('Vrai Lung : ', (0.78*DictInterpolated['Lung'][0][index])+(0.22*DictInterpolated['Air'][0][index]),'\n\n')
    phantom_muAtt[energy]=np.where(phantom_muAtt[energy]==DictInterpolated['Lung'][0][index],
                                   (0.78*DictInterpolated['Lung'][0][index])+(0.22*DictInterpolated['Air'][0][index]),
                                   phantom_muAtt[energy])
    
    print('muEN')
    print('Energy : ',energy)
    print('Lung : ',DictInterpolated['Lung'][1][index])
    print('Air : ',DictInterpolated['Air'][1][index])
    print('Vrai Lung : ', (0.78*DictInterpolated['Lung'][1][index])+(0.22*DictInterpolated['Air'][1][index]),'\n\n')
    
    phantom_muEn[energy]=np.where(phantom_muEn[energy]==DictInterpolated['Lung'][1][index],
                                    0.78*DictInterpolated['Lung'][1][index]+0.22*DictInterpolated['Air'][1][index],
                                    phantom_muEn[energy])
    
phantom_density = np.where(phantom_density==LungDensity,0.35,phantom_density)

# %% [markdown]
# ### **Question 11.1**
# 
# 11. Calculer et tracer la variation de la dose due au rayonnement primaire en profondeur selon l’axe en pointillé.

# %% [markdown]
# # Flux et dose avec nouveau poumon et ancienne fluence

# %%
# Fiste Calculate the new fluence matrix
test = np.where(DistanceArray != 0, 1, 0)
NewMatrixFlux3 = MatriceOfFLuxV2(phantom_muAtt, phantom_density, flux, test ,EnergyValues)



# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(NewMatrixFlux3[i,:,:]-MatriceOfFlux2[i,:,:])
    plt.title('Matrix of flux at ' + str(EnergyValues[i]) + ' MeV')
    plt.colorbar()
plt.show()

# %% [markdown]
# La soustraction des flux montre clairement que le flux est plus élevé dans le cas du vrai poumon. Cette différence est principalement due au fait que le coefficient d'atténuation massique dans le vrai poumon est plus bas, comme illustré dans la fonction précédente. De plus, la densité du vrai poumon est significativement plus faible, ce qui amplifie la différence.
# 
# Cependant, il est important de noter que le coefficient d'absorption massique est presque similaire entre les deux situations. Cela signifie que malgré des différences marquées dans l'atténuation des rayonnements, la proportion de rayonnements absorbés dans le vrai poumon reste comparable. Cette observation suggère que les doses déposées dans le vrai poumon seront probablement plus élevées. ( c'est une approximation forte et non valide selon moi )

# %% [markdown]
# ___

# %%
Dose_axe_New_flux = Compute_dose_depth_axe(phantom_muAtt,phantom_muEn,phantom_density,NewMatrixFlux3,EnergyValues,DistanceArray)


# %%

Dose_axe_New_flux = np.asarray(Dose_axe_New_flux)
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.plot(Dose_axe_New_flux[i,:],label = str(EnergyValues[i])+' MeV new flux')
    plt.plot(Dose_axe[i,:],linestyle='--' ,label = str(EnergyValues[i])+' MeV old flux')
    plt.yscale('log')
    plt.legend()
    plt.title('Dose depth curve')
    plt.xlabel('Depth')
    plt.ylabel('Dose ')
plt.show()

# %% [markdown]
# # Flux et dose avec nouveau poumon et nouvelle fluence

# %%
# Fiste Calculate the new fluence matrix
test = np.where(DistanceArray != 0, 1, 0)
NewMatrixFlux2 = MatriceOfFLuxV2(phantom_muAtt, phantom_density, flux_entre, test ,EnergyValues)

# %%
Dose_axe_New_flux = Compute_dose_depth_axe(phantom_muAtt,phantom_muEn,phantom_density,NewMatrixFlux2,EnergyValues,DistanceArray)

# %%
Dose_axe_New_flux = np.asarray(Dose_axe_New_flux)

plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.plot(Dose_axe_New_flux[i,:],label = str(EnergyValues[i])+' MeV')
    plt.yscale('log')
    plt.legend()
    plt.title('Dose depth curve')
    plt.xlabel('Depth')
    plt.ylabel('Dose ')
plt.show()

# %% [markdown]
# ## **Question 11.3**
# 
# 11. Calculer les cartes de dose primaire aux quatre énergies.

# %%
Dose_Map_New_flux= DoseMapv2(phantom_muAtt,phantom_muEn,phantom_density,NewMatrixFlux2,EnergyValues,DistanceArray)

# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Dose_Map_New_flux[i,:,:],cmap='jet')
    plt.colorbar()
    plt.title('Dose map for '+str(EnergyValues[i])+' MeV')
plt.show()

# %% [markdown]
# ## **Question 12**
# 
# 12. Sachant qu'on détecte 3000 photons sur le pixel situé au regard de l'axe du faisceau, tracez le profil d'intensité obtenu sur un détecteur pixelisé placé en aval du patient (pixels de 1mm) pour chaque faisceau. Calculez le contraste sur le profil d'intensité entre deux points situés à 1 mm de part et d'autre de l'axe du faisceau

# %%
# First we calculate the matrix of flux using the function MatriceOfFLuxV2
test = np.where(DistanceArray != 0, 1, 0)
NewMatrixFluxAxis = MatriceOfFLuxV2(phantom_muAtt, phantom_density, flux_entreaxial, test ,EnergyValues)

# %%
Dose_Map_New_flux_axis= DoseMapv2(phantom_muAtt,phantom_muEn,phantom_density,NewMatrixFlux2,EnergyValues,DistanceArray)

# %%
plt.figure(figsize=(20, 15))
for i in range (len(EnergyValues)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Dose_Map_New_flux_axis[i,:,:],cmap='jet')
    plt.colorbar()
    plt.title('Dose map Normalized for '+str(EnergyValues[i])+' MeV')
plt.show()

# %%
profil = []

for i in range (len(EnergyValues)):
    #Just to check if the fluence have the value we expected
    print(NewMatrixFluxAxis[i,900,-1])
    # ici reshape (-1,10) -1 commande a numpy de calcule automatiquement le nombre de ligne pour avoir 10 colonnes et garder le meme nombre de valeur
    # np.mean calcule la moyenne de chaque ligne
    # ainsi ici on prend la dernier colone de la matrice de dose et on la reshape dabs notre cas le nombre de ligne est de 1800 on aura donc 180 ligne et 10 colonne
    # on calcule la moyenne de chaque ligne et on obtient donc 180 valeur
    tmp = np.mean(Dose_Map_New_flux_axis[i,:,-1].reshape(-1,10),axis=1)
    plt.plot(tmp/np.max(tmp),label = str(EnergyValues[i])+' MeV')
plt.legend()
plt.title('Dose profile normalized ')
plt.xlabel('Lateral distance')
plt.ylabel('Dose normalized')
plt.show()
    
    


