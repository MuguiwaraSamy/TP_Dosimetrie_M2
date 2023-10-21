## TP1

#### **Travail à réaliser:**  

1. Expliquez ce qu’est une **courbe de survie**.  

2. A  partir  des  4  courbes  de  la  publication,  **récupérer  graphiquement**  les  valeurs  des  mesures {dose, survie et incertitude sur la survie} et les sauvegarder au format excel. 

3. **Ouvrir** dans python les données excel  et **retrouvez graphiquement** l’allure des mesures de la publication.  Vous  utiliserez  une  échelle  semi-logarithmique  et  vous  afficherez  les  incertitudes  de  mesure. 

4. Tracer  dans  un  **graphique**  la  survie  des  deux  types  de  cellules  pour  l’irradiation  avec  les  photons.  Commentez. 

5. Refaites de même pour l’irradiation avec les neutrons. 

*On va modéliser la tendance des données avec le modèle linéaire quadratique $S = e^{-\alpha D - \beta D^{2}}$ On peut avoir deux approches: ajuster directement avec la fonction exponentielle décroissante à 2 paramètres, ou encore ajuster le logarithme népérien de la surivie par un polynôme d’ordre 2.* 

6. **Ajuster** les courbes de survie avec les 2 approches et affichez les graphiquement.  

7. **Comparer** les résultats : valeurs des paramètres $\alpha$ et $\beta$ et précision obtenue. 

8. Expliquer comment marche la fonction **curve_fit** de Python. 

9. Evaluer la qualité de votre fit avec la valeur du **$\chi^{2}$**. Commentez.  

10. Calculez le rapport **$\alpha / \beta$** et l’incertitude associée grâce à la **matrice de covariance** pour chaque type d’irradiation et de lignée cellulaire.   

11. Comparez  tous  les  **paramètres  de  radiosensibilité  $\alpha$ et  $\beta$** et  le  rapport **$\alpha / \beta$** que  vous  avez  obtenus  à  ceux  de  la  publication.  Commentez  quant  à  la  radiosensibilité  de  ces  cellules  aux  
différentes irradiations.  

12. On considère la lignée cellulaire WSU-DLCL2. A partir des courbes que vous avez obtenues à 
la question 6, déterminez par interpolation la **dose** à appliquer avec des photons, puis des neutrons 
pour avoir l’effet biologique suivant: 
* 1 % de survie
* 10 % de survie
* 50 % de survie
* 80 % de survie
* 90 % de survie

13. Expliquer ce qu’est **l’efficacité biologique relative** (EBR en français ou RBE dans la publi).  

14. **Calculer** l’EBR neutrons pour chacune des lignées cellulaires et representez le résultat sous forme **d’un même graphique**. Vous tracerez l’EBR des neutrons en fonction de la survie de 1 à 95 % par pas de 1%. Commentez les résultats.  

15. En radiobiologie, on peut réaliser un test simple qui ne nécessite pas la réalisation de courbes de survie cellulaires complètes. Il s’agit du calcul du **rapport des survies à 2 Gy**. Calculer le rapport des survies à 2 Gy dans notre cas. Commenter la pertinence de cet indicateur radiobiologique dans le cas de la comparaison des irradiations neutron et gamma sur ce type de cellules. 