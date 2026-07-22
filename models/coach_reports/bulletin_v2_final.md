# Bulletin de fin de scolarité

**Élève :** v2 finale
**Date :** 22/07/2026
**Entraînement :** 28 M d'itérations (fin de scolarité)
**Mode de notation :** moyenne purgée (note haute et note basse retirées, puis moyenne des restantes)

---

## Notes par matière

| Matière | Note /10 | Appréciation |
|---|---|---|
| Préflop | 7,5 | Le vrai leak de la v1 est corrigé : premiums quasi jamais jetés (BB de 10,6 % à 2,4 % de fold), ranges élargies conformes au HU et 3-bet enfin polarisé/mixte. Restent un sur-limp des premiums au bouton et un fold BB « bonne » (38 %) trop tight face aux petites relances. |
| Tirages | 8,5 | Postflop mûr : cbet flop dans la zone saine (58,5 %), mix tirage/main-faite quasi indistinguable à la relance (25,8 % vs 28,3 %), agression river rééquilibrée (~2:1). Léger soupçon de sur-fold flop (52 %) à surveiller. |
| River | 8,0 | Ratio value:bluff de 2,11:1, presque pile le 2:1 théorique d'inexploitabilité pour une mise pot. Bluffs bien polarisés sur tirages manqués. Échantillon mince (n=37) et bucket « entre_deux » (24 %) encore ambigu. |
| Discipline | 7,0 | Défense nettement plus mûre et mixée (check-raise flop 28 % vs 13 % en v1), folds face au tapis corrects et GTO-défendables. Mais un trou d'over-fold aux bas prix subsiste : fold indéfendable de TT préflop pour 0,5 BB, et couchers réflexes face aux mini-mises. |

**Moyenne générale (purgée) : 7,75 / 10**
*(notes retenues : 7,5 et 8,0 ; note haute 8,5 et note basse 7,0 écartées — la moyenne simple donne également 7,75)*

---

## Appréciation du directeur

La v2 a atteint la maturité stratégique que l'on attendait d'un blueprint à 28 M d'itérations : le style s'est équilibré, les ranges sont devenues mixtes et polarisées, les semi-bluffs et le check-raise ont remplacé la passivité, et la river est passée d'un profil scolaire et exploitable (4,43:1 en v1) à un ratio quasi inexploitable (2,11:1). C'est un vrai gain d'équilibre et de protection de range, pas un simple gain d'équité brute. Mais soyons honnêtes : malgré cette montée en finesse, la v2 reste statistiquement à égalité mesurée avec la v1 (3 M) — l'abstraction plus fine a amélioré le STYLE et l'illisibilité sans creuser d'écart de win-rate mesurable. Il subsiste par ailleurs un défaut de discipline aux bas prix (fold de premium préflop pour 0,5 BB, sur-fold face aux mini-mises) qui reste directement exploitable et doit être corrigé en priorité. Bilan : une élève accomplie et bien équilibrée, mais dont les progrès sont qualitatifs plus que quantitatifs — pas de survente.

---

## Verdict pour le match contre l'humain semi-pro

Prête à jouer et difficile à lire : son équilibre river et son jeu mixé la protègent contre un adversaire qui cherche à l'exploiter frontalement. Face à un semi-pro, elle ne se fera pas prendre sur les gros sizings — sa discipline y est mûre. Le seul angle d'attaque réel est le sur-fold aux petites tailles (≤ 1/3 pot) et le fold de premiums face aux mini-relances : un humain observateur peut les cibler à bas prix. À corriger avant le match, mais le profil global tient la route.
