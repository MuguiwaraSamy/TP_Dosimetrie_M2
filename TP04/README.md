# Master Ingénierie de la Santé / Master Physique
## Physique Médicale, Radioprotection de l’homme et de l’environnement
**Université Grenoble-Alpes 2023-2024**

### Travaux Pratiques de modélisation pour la dosimétrie
#### TP #5 & 6: Initiation au calcul Monte Carlo

**Contacts:**
- Véronica Sorgato: [veronica.sorgato88@googlemail.com](mailto:veronica.sorgato88@googlemail.com)
- Samy Kefs: [samy.kefs@inserm.fr](mailto:samy.kefs@inserm.fr)
- Yannick Arnoud: [yannick.arnoud@lpsc.in2p3.fr](mailto:yannick.arnoud@lpsc.in2p3.fr)

**Données:**
Les données de coefficients d’atténuation et d’absorption en énergie de l’eau sont données dans le tableau «eau.xlsx». Il s’agit des données extraites de la base de données du NIST.
On considère des faisceaux de photons parallèles de 20 keV, 140 keV, 6 MeV et 18 MeV. On considère une cuve à eau (cf TP 2).

### Travail à réaliser:

1. À partir d’un nombre aléatoire R compris entre 0 et 1 et uniformément réparti, écrire le code pour générer un nombre aléatoire uniformément réparti entre m (valeur minimale) et M (valeur maximale).

2. À partir d’un nombre aléatoire R compris entre 0 et 1 et uniformément réparti, écrire le code permettant de décider le tirage d’un dé à 6 faces.

3. On connaît les valeurs de $\mu_{pe}=0,72 cm^{-1}$, $\mu_{Rayleigh}=0,01 cm^{-1}$, et $\mu_{Compton}=0,57 cm^{-1}$ pour un photon d’énergie inférieure à 1,022 MeV. À partir d’un nombre aléatoire R compris entre 0 et 1 et uniformément réparti, écrire le code permettant de décider quelle interaction va suivre le photon.

4. Écrire le code Monte Carlo permettant d’estimer le nombre $\pi$ à partir de 2 nombres aléatoires tirés de manière uniforme entre 0 et 1. On pourra essayer de visualiser le carré et le cercle tels que vus en cours. Commentez chaque section de votre code (en annexe) et expliquer la formule qui estime $\pi$.

5. **Bonus:** Faire varier le nombre de tirages et commenter les résultats obtenus.

6. **Bonus:** À partir de quand avons-nous un écart relatif inférieur à 1/1000, inférieur à 1/100000. Mesurez le temps nécessaire pour obtenir ces incertitudes.

7. Écrire le code Monte Carlo permettant de simuler l’émission isotrope d’une source. Pour obtenir l’angle azimutal θ, vous utiliserez la méthode de rejet (bonus: la méthode d’inversion de la fonction de répartition). Représentez le passage des photons par la surface d’une sphère (histogramme 3D) de rayon 1m.

8. On souhaite collimater la source de telle façon que les photons de la source isotrope soient émis selon un cône. La génératrice de ce cône est l’axe 0z de la sphère, et son angle au centre θ vaut 10°, ce qui correspond à une ouverture de 20°. Écrire et commentez le code (en annexe) permettant de réaliser cette simplification.
![Exemple][Picture.jpg]

9. On considère maintenant la propagation de photons dans une cuve à eau. Calculez par la méthode de Monte Carlo la distance parcourue dans l’eau avant interaction par chaque photon:
   - Méthode élémentaire (dx)
   - Méthode d’inversion de la fonction de répartition.
   On fera le calcul pour chaque énergie de faisceau.

10. En déduire le coefficient d’atténuation de l’eau simulé (fit exponentiel).

11. Faire 1000x la simulation en déduire la répartition de mu et le biais éventuel. Commentez.

12. Déterminez le type d’interaction au point d’interaction.

13. Calculez l’énergie moyenne transférée aux particules chargées. Calculez mutr/ho de l’eau. Comparez aux données du TP 3 et commentez.
