# Bulletin scolaire — École de Poker

**Élève :** v2 (IA poker CFR)
**Date :** 22/07/2026
**Jalon :** 225 000 itérations sur 3 000 000 (7,5 % de la scolarité)
**Référence :** v1 finale (3M itérations)

---

## Notes par matière

| Matière | Note /10 | Appréciation |
|---|---|---|
| Préflop | 3,5 | Bonne ouverture au bouton, proche de la théorie, mais la big blind est un bruit non convergé : 3-bet massif avec les mains faibles et folds de premiums inacceptables. |
| Tirages | 6,5 | Excellente surprise : la cécité aux tirages de la v1 est corrigée, cbet dans la norme, semi-bluff acquis — reste à rendre l'agressivité sélective. |
| River | 3 | Le bluff existe et les sizings varient, mais sur-bluff sévère (1,3:1 au lieu de 2:1) avec des all-in à équité quasi nulle : le déséquilibre le plus coûteux. |
| Discipline | 3 | Structure défensive inversée : relance ses poubelles, jette ses monstres face à des petites mises. Raise-mania (47 % flop, 65 % river) à corriger d'urgence. |

## Moyenne générale : **4,0 / 10**

---

## Appréciation du directeur

v2 est un jeune élève turbulent mais prometteur. À seulement 7,5 % de sa scolarité, il a déjà accompli ce que son aîné n'a jamais réussi en trois millions d'itérations : voir ses tirages et les jouer avec agressivité — un progrès clé qui laisse entrevoir un plafond supérieur à celui de la v1. Mais cette fougue est encore brute : il relance tout, bluffe trop la river et couche ses monstres, ce qui se paie au prix fort (-158 bb/100 contre +281 pour la v1). Le talent est là ; il faut maintenant laisser le travail (l'entraînement CFR) transformer cette énergie en jugement.

---

## Objectifs pour le prochain jalon

1. **Big blind (préflop)** — priorité absolue : concentrer le 3-bet sur les premiums, basculer les mains faibles/marginales vers call ou fold, et cesser tout fold de premium.
2. **River** — faire descendre le bluff sous ~25 % et le raise face à une mise river sous 40 % ; viser un ratio value:bluff proche de 2:1 sur les mises pot.
3. **Discipline face à une mise** — viser ~50-55 % de fold et moins de 20 % de raise flop/turn, en corrigeant le tri : ne jamais folder 0,70+ d'équité face à une mise inférieure au demi-pot.
4. **Tirages** — préserver l'acquis : maintenir l'écart raise tirage > main faite pendant que le raise global redescend, et défendre les combos tirage+paire au lieu de les folder.
