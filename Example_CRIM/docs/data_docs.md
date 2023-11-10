# Explication rapide de chaque colonne dans les fichiers csv

Colonnes:
- `AcquisitionTime`: Temps d'acquisition de chaque lecture. Comme les lectures sont en "batch", il y a toujours plusieurs lectures faites très rapprochées (à quelques secondes d'intervalles), avec des `config` différentes.
- `DeviceId`: Identification de chaque senseur, 1 par silo.
- `temperature`: Température (celsius) mesurée sur le senseur.
- `config`: Nom de la configuration pour la lecture.
- `raw_data`: Le signal ultrason brut, sous forme d'une liste d'entier entre 0 et 255.
- `pulseCount`: Nombre de pulse envoyé par le capteur, dépend de la `config`.
- `batchId`: Identificateur unique pour chaque "batch" de lectures, qui permet donc de les identifier facilement.
- `samplingFrequency`: Fréquence d'acquisition du senseur, dépend de la `config`. Présentement tous les senseurs ont la même valeur, donc peu d'utilité comme valeur.
- `decimationFactor`: Relié à la fréquence d'acquisition, tous les senseurs ont la même valeur donc peu intéressant.
- `LocationName`: Nom attribué à chaque silo. Il y a présentement 10 silos dans ce dataset qui sont reliés à des balances.
- `h1` `h2` `h3` `LT` `AngleDuCone` `AngleDuToit` `Diametre`: Dimensions du silo (en mètres et degrés), tel que la documentation.
- `siloDimensions`: Tous les détails de dimensions du silo, sous forme de `dict`
- `mainBangEnd`: Fin du bang d'émission calculé par l'algo.
- `sensorWavefront`: Valeur de front d'onde calculé par l'algo. C'est cette valeur qui nous intéresse au final et qui permet de déterminer la distance entre le senseur et le grain, puis le poids du grain restant.
- `signalQuality`: Qualité de l'onde telle que calculé par l'algo. Une valeur plus haute indique une onde de meilleure qualité.
- `siloHeight`: Hauteur du silo en mètres.
- `trueWeight`: Poids (en tonnes) du grain restant mesuré par un "Loadcell" (une balance), considéré comme la vraie valeur.
- `trueWavefront`: Valeur du front d'onde associé à la mesure de la balance, après la transformation du poids -> distance
