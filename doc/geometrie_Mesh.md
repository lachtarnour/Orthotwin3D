# Normal a un point 
Pour calculer la normale d’un point Pi dans un mesh, on récupère d’abord les triangles adjacents à ce point. Ensuite, on calcule la normale de chaque triangle avec le produit vectoriel de deux arêtes. Enfin, on fait une moyenne des normales des triangles voisins, souvent pondérée par l’aire de chaque triangle, puis on normalise le vecteur obtenu pour avoir une normale unitaire.

# Calcule de courbure (differenciel)
## approximation par polynome 
Pour calculer les paramètres différentiels autour d’un sommet Pi, on commence par estimer sa normale ni, puis on définit le plan tangent Π passant par Pi et de normale ni.

On considère ensuite l’ensemble des sommets voisins Pj ∈ N(Pi). Chaque voisin Pj est exprimé dans un repère local centré en Pi, composé de deux axes tangents u et v, et de l’axe normal ni.

Pour chaque voisin Pj, on calcule donc ses coordonnées locales (uj, vj, wj), où uj et vj représentent sa position dans le plan tangent, tandis que wj représente sa hauteur signée par rapport à ce plan.

L’objectif est alors d’approximer localement la surface du maillage par une surface quadratique :

w(u, v) = au² + 2buv + cv²

Pour chaque voisin Pj, la hauteur prédite par ce modèle est :

w(uj, vj) = auj² + 2bujvj + cvj²

On cherche donc les coefficients a, b et c qui minimisent l’erreur entre la hauteur prédite par le modèle et la hauteur réelle wj :

E = Σ [(auj² + 2bujvj + cvj²) - wj]²

Cette minimisation se fait par moindres carrés. Une fois les coefficients a, b et c estimés, on obtient une surface continue locale qui approxime le voisinage de Pi. À partir de cette surface quadratique, on peut ensuite calculer les courbures principales k1, k2 et leurs directions principales associées.


## approximation par gaussienne 
Autour d’un sommet Pi, les triangles voisins forment une sorte de cône.
Si les angles autour de Pi font exactement 360°, la zone est plate.
S’il manque de l’angle, la zone est convexe.
S’il y a trop d’angle, la zone est en selle.
Donc :
déficit d’angle = 2π - somme des angles autour de Pi
courbure gaussienne ≈ déficit d’angle / aire locale

