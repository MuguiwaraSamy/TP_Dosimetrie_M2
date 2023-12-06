# BIENVENUE A VOTRE DERNIERE SEANCE DE TP DE PYTHON ET DOSIMETRIE
# Aujourd'hui, nous allons faire un peu de calcul de dose, quelques fits, et un peu de statistiques
# Nous allons utiliser les modules numpy, scipy, matplotlib et pandas
# Je vous conseille de garder la documentation de ces modules ouverte dans un navigateur
# pour pouvoir vous y référer facilement.

# Essayer de ne pas utiliser CHAT-GPT pour ce TP, commentez votre code, et essayez de comprendre
# ce que vous faites. 
# Le but est de vous faire manipuler des outils de calcul et de statistiques, pas de faire du copier-coller
# de code. Mais aussi de vous faire comprendre les bases de la dosimétrie. 

# Les diffrentes questions sont a faire en groupes, ainsi il faut cree des fonction pour chaque question

################################################################################################################################
# Question 1 :
# Nous allons commencer par importer les modules dont nous aurons besoin
# Vous allez devoir importer les modules numpy, matplotlib.pyplot et pandas mais aussi la fonction curve_fit de scipy.optimize
# temps estimé : 1 minutes
################################################################################################################################


''' VOTRE CODE ICI '''

################################################################################################################################
# POUR CHAQUE QUESTION CREE UNE NOUVELLE FONCTION QUE TU APPELLERAS DANS LE MAIN
# UTILISE LA FONCTION IF __NAME__ == "__MAIN__" POUR APPELER TES FONCTIONS
################################################################################################################################




################################################################################################################################
# Question 2 :
# Nous allons maintenant importer les données de la feuille de calcul Excel Data_Water.xlsx
# ( N'hesitez pas à l'ouvrir pour voir à quoi elle ressemble )
# temps estimé : 5 minutes
################################################################################################################################


''' VOTRE CODE ICI '''


################################################################################################################################
# Question 2.1 :
# representez les données de la feuille de calcul sous forme de graphique 
# Afficher chaque colone sur un graphique différent et un dernier graphique avec les courbes pertinente sur le même graphique
# N'oubliez pas de mettre des titres et des légendes sur vos graphiques
# temps estimé : 5 minutes
################################################################################################################################


''' VOTRE CODE ICI '''

################################################################################################################################
# Question 2.2 : 
# Faite une interpolation linéaire des données de la feuille de calcul pour 10kev  
# Vous pouvez passer en espace log-log pour faire votre interpolation (log naturel) 
# temps estimé : 10 minutes
################################################################################################################################

''' VOTRE CODE ICI '''


################################################################################################################################
# Question 2.3 :
# Le TERMA vaut a la surface de l'eau 1.5 Gy
# Calculer la fluence en energie et en nombre de particule pour 10 kev
# En deduire le KERMA 
# temps estimé : 10 minutes
################################################################################################################################

''' VOTRE CODE ICI '''

################################################################################################################################
# Question 3 :
# Simuler le point d'entré d'un faisceau de 10 kev collimaté de rayon 10.5 cm à 1m de distance 
# N'oubliez pas de le représenter graphiquement
# ( Quelques lignes de code suffisent ) 
# temps estimé : 10 minutes
################################################################################################################################

''' VOTRE CODE ICI '''

################################################################################################################################
# Question 4 :
# Nous allons maintenant simuler la propagation de ce faisceau dans l'eau
# Trouvez la distance d'intéraction de chaque particule dans l'eau
# N'oubliez pas de le représenter graphiquement
# temps estimé : 10 minutes
################################################################################################################################

''' VOTRE CODE ICI '''

################################################################################################################################
# Question 5 :
# Nous allons maintenant simuler le type d'interaction de chaque particule dans l'eau
# Trouvez le type d'interaction de chaque particule dans l'eau
# printez le nombre de chaque type d'interaction ou representez le graphiquement
# temps estimé : 10 minutes
################################################################################################################################


''' VOTRE CODE ICI '''


################################################################################################################################
# Question 6 : 
# il se trouve que nous avons irradié des cellules voici les donné de survie de ces cellules
# dans le fichier xlsx
# importez les données et représentez les graphiquement
# temps estimé : 5 minutes
################################################################################################################################


''' VOTRE CODE ICI '''  

################################################################################################################################
# Question 7 :
# Nous allons maintenant faire un fit de ces données
# faire un fit avec le MLQ 
# temps estimé : 5 minutes
################################################################################################################################
    
    
''' VOTRE CODE ICI '''