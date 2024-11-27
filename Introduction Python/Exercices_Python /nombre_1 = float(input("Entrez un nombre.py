nombre_1 = float(input("Entrez un nombre : "))
nombre_2 = float(input("Entrez un autre nombre : "))
operateur = input("Entrez un opérateur (+, -, *, /) : ")

if operateur == "+":
    print("Le résultat de l'addition est", nombre_1 + nombre_2)
elif operateur == "-":
    print("Le résultat de la soustraction est", nombre_1 - nombre_2)
elif operateur == "*":
    print("Le résultat de la multiplication est", nombre_1 * nombre_2)
elif operateur == "/":
    if nombre_2 == 0:
        print("Division par zéro impossible")
    else:
        print("Le résultat de la division est", nombre_1 / nombre_2)
else:
    print("Opérateur inconnu")