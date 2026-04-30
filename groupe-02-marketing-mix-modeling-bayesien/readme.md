EQUIPE
Lans Léo
Esnault Wandrille
Jezequel Martin 

Projet : Description : Un sujet tres demande en entreprise : optimiser le budget publicitaire. Le Marketing Mix Modeling (MMM) attribue les ventes aux differents canaux (TV, Facebook, Google) en tenant compte des effets de saturation (rendements decroissants) et de delai temporel (Adstock). L'approche bayesienne avec PyMC permet d'estimer ces parametres inconnus avec quantification d'incertitude, et de simuler des scenarios d'allocation optimale.

Objectifs gradues :

Minimum : Modele lineaire bayesien simple avec PyMC, 2-3 canaux, estimation des coefficients
Bon : Modele hierarchique avec effets de saturation (Hill transform) et Adstock, optimisation budget, visualisation des contributions
Excellent : Modele multi-marche hierarchique, validation croisee temporelle, comparaison avec Google LightweightMMM, simulation de scenarios
Notebooks de reference :

Notebook	Description	Lien
Infer-101	Introduction inference bayesienne	Infer-101
Note : Ce sujet utilise principalement PyMC (Python). Le notebook Infer-101 fournit les bases bayesiennes ; les tutoriels PyMC-Marketing sont le vrai point de depart.

References externes :

PyMC-Marketing - MMM bayesien avec PyMC (point de depart recommande)
PyMC-Marketing MMM Tutorial - Tutoriel complet pas a pas
Google LightweightMMM - Implementation Google (pour comparaison)
Bayesian Methods for Media Mix Modeling (Jin et al.) - Paper de reference