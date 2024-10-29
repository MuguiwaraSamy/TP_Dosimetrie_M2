# Travaux Pratiques de Modélisation pour la Dosimétrie

## TP #2 : Étude des rendements en profondeur dans l'eau

### Contacts :
- Verónica Sorgato: veronica.sorgato88@gmail.com
- Samy Kefs : samy.kefs@inserm.fr
- Yannick Arnoud: yannick.arnoud@lpsc.in2p3.fr

### Données :
- À partir du site du NIST, récupérer dans un fichier Excel les coefficients d'atténuation $\mu_{att}$ et d'absorption en énergie $\mu_{en}$ du PMMA (Polyméthyl Méthacrylate) en fonction de l'énergie des photons, ainsi que ca masse volumique.

### Travail à Réaliser :

1. Ouvrir le fichier Excel et afficher les valeurs discrètes $\mu_{att}$ et $\mu_{en}$ en fonction de l'énergie sur une courbe en échelle doublement logarithmique en x et en y (log-log). 
2. Ajuster ces points par un modèle linéaire et afficher sur le même graphe cette courbe d'ajustement. Commenter.
3. Tester d'autres types d'ajustement. Au vu des graphiques obtenus, êtes-vous satisfaits du résultat ? Vous pouvez prendre comme référence d'une courbe au comportement « sain » celle affichée sur le site NIST.
4. Essayer l'interpolation linéaire en échelle logarithmique et tracer là. Est-ce que vos résultats sont meilleurs ?
5. Le site internet de questions réponses **stackoverflow** propose une fonction d’ajustement qui a retenu notre attention. Qu’en pensez-vous ?
https://stackoverflow.com/questions/29346292/logarithmic-interpolation-in-python
6. Avec votre meilleur ajustement, trouver les valeurs $\mu_{att}$ et $\mu_{en}$ pour des énergies de faisceaux de photons de 20 keV, 140 keV, 6 MeV et 18 MeV.



### Questions Suivantes :

**Pour les prochaines questions, on considère une fluence de photons par cm2 mesurée à 1m de la source. On dispose d’un fantôme de PMMA de 1m de hauteur, dont la surface est placée à 1m de la source.**

6. Calculer la dose absorbée à l'équilibre électronique à la surface du fantôme de PMMA de 1 mètre de hauteur, Commenter.
7. Calculer et afficher les valeurs de la dose en fonction de la profondeur pour chacun des 4 faisceaux étudiés, en considérant un faisceau parallèle. Commenter.
8. Calculer et tracer la dose en fonction de la profondeur (pas de 1 mm), en normalisant au maximum de dose, puis tracer sur le même graphe les quatre rendements en profondeur de la dose absorbée due aux photons primaires. Commenter au regard des rendements en profondeur vus en cours ou trouvés sur internet.
9. Si on insère une chambre d'ionisation à 50 cm de profondeur dans le fantôme PMMA, quelle serait la mesure de dose attendue ?
10. Pour une application en radioprotection, souhaitant avoir une fluence en sortie du fantôme de 1 photon/mm², calculer la dose en entrée correspondante, ainsi que la dose à 10 cm de profondeur. Commenter.