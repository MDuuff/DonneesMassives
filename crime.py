# coding=utf-8

# Pour le chargement en mémoire des fichiers de données au format CSV
import pandas as pd

# Precision score pour chaque categorie
from sklearn.metrics import average_precision_score

# Preprocessing permet de transformer les noms des catégories de crimes
# en valeurs (nombres entiers) uniques
from sklearn import preprocessing

import numpy as np

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB

# Pour splitter les données d'entraînerment pour apprentissage du classificateur
from sklearn.cross_validation import train_test_split

# Pour comparer les classificateurs et déterminer lequel prédit avec le moins de pertes de connaissances
# Utilisation du LOG LOSS
from sklearn.metrics import log_loss

# ALGORITHME

# Chargement des données
testCsv = pd.read_csv('Data/test.csv', low_memory=False, parse_dates = ['Dates'])
trainCsv = pd.read_csv('Data/train.csv', low_memory=False, parse_dates = ['Dates'])

# Suppression des valeurs aberrantes (Longitude = 90 degrés)
trainCsv = trainCsv[trainCsv.Y != 90]

# Décomposition de l'attribut Dates en attributs temporels
year = trainCsv.Dates.dt.year
month = trainCsv.Dates.dt.month
hour = trainCsv.Dates.dt.hour

# Transformation des valeurs des attributs temporels en nombres binaires
# pour la compatibilité avec le classificateur
# Fonction get_dummies() de pandas.
year = pd.get_dummies(year)
month = pd.get_dummies(month)
day = pd.get_dummies(trainCsv.DayOfWeek)
hour = pd.get_dummies(hour)

# Transformation des valeurs des valeurs des attributs
# nominaux ordinaux en valeurs binaires
district = pd.get_dummies(trainCsv.PdDistrict)

# Division de la plage de valeurs des attributs Latitude et Longitude
# en 21 intervalles de taille égale
cutXList = [-122.513642, -122.50679004761905, -122.4999380952381, -122.49308614285714, -122.48623419047618, -122.47938223809523, -122.47253028571427, -122.46567833333332, -122.45882638095236, -122.4519744285714, -122.44512247619045, -122.43827052380949, -122.43141857142854, -122.42456661904758, -122.41771466666663, -122.41086271428567, -122.40401076190471, -122.39715880952376, -122.3903068571428, -122.38345490476185, -122.37660295238089, -122.36975099999994]
cutYList = [37.707879, 37.71324766666667, 37.71861633333334, 37.723985000000006, 37.729353666666675, 37.734722333333345, 37.740091000000014, 37.74545966666668, 37.75082833333335, 37.75619700000002, 37.76156566666669, 37.76693433333336, 37.77230300000003, 37.7776716666667, 37.78304033333337, 37.78840900000004, 37.793777666666706, 37.799146333333375, 37.804515000000045, 37.809883666666714, 37.81525233333338, 37.82062100000005]

# Puis transformation des intervalles obtenus en nombres binaires
cuttedX = pd.cut(trainCsv.X, cutXList)
cuttedXDummie = pd.get_dummies(cuttedX)
cuttedY = pd.cut(trainCsv.Y, cutYList)
cuttedYDummie = pd.get_dummies(cuttedY)

# Création d'un nouveau Dataframe contenant les attributs que
# nous venons de créer, de façon horizontale (option axis=1)

# new_train = pd.concat([district, year, month, day, hour, cuttedXDummie, cuttedYDummie], axis=1)

# Sélection finale des attributs à prendre en compte dans l'apprentissage du modèle
# cuttedXDummie = Latitude
# cuttedYDummie = Longitude
new_train = pd.concat([hour, cuttedXDummie, cuttedYDummie], axis=1)


##############################
# Même travail sur le fichier de tests
##############################
year = testCsv.Dates.dt.year
month = testCsv.Dates.dt.month
hour = testCsv.Dates.dt.hour

year = pd.get_dummies(year)
month = pd.get_dummies(month)
day = pd.get_dummies(testCsv.DayOfWeek)
hour = pd.get_dummies(hour)
district = pd.get_dummies(testCsv.PdDistrict)

cuttedX = pd.cut(testCsv.X, cutXList)
cuttedXDummie = pd.get_dummies(cuttedX)
cuttedY = pd.cut(testCsv.Y, cutYList)
cuttedYDummie = pd.get_dummies(cuttedY)

# Les différents scénarios de tests en ce qui concerne la combinaison d'attributs sélectionnée
# La combinaison choisie est décommentée

# CAS AVEC TOUS LES ATTRIBUTS PRE-SELECTIONNES
# new_test = pd.concat([district, year, month, day, hour, cuttedXDummie, cuttedYDummie], axis=1)

# CAS AVEC TOUS LES ATTRIBUTS SAUF LES JOURS, MOIS ET ANNEES
# new_test = pd.concat([district, day, hour, cuttedXDummie, cuttedYDummie], axis=1)

# CAS AVEC TOUS LES ATTRIBUTS SAUF LES JOURS, MOIS, ANNEES ET COORDONNEES GEOGRAPHIQUES
# new_test = pd.concat([district, hour], axis=1)

# CAS AVEC TOUS LES ATTRIBUTS SAUF ELEMENTS TEMPORELS
# new_test = pd.concat([district, cuttedXDummie, cuttedYDummie], axis=1)

# CAS SANS LES COORDONNEES GEOGRAPHIQUES
# new_test = pd.concat([district, year, month, day, hour], axis=1)

# CAS AVEC HEURE ET COORDONNES LATITUDE ET LONGITUDE
new_test = pd.concat([hour, cuttedXDummie, cuttedYDummie], axis=1)

# Transformation de chaque catégorie de crime en un identifiant
# numérique (ID) unique
encoder = preprocessing.LabelEncoder()
numcatego = encoder.fit_transform(trainCsv.Category)

# Ajout de cet attribut au dataFrame récemment créé
new_train['Numcatego'] = numcatego

# Pour obtenir la performance (log loss) du model choisi:
# Réglage à 80% de la proportion de données d'entraînement
train, perf = train_test_split(new_train, train_size=.80)

# Liste des noms des colonnes du Dataframe créé à partir des données
discrete = list(new_test)


# On instancie le modèle Multinomial du classificateur naïf de Bayes
nb = MultinomialNB()


####### PARTIE CALCUL DU LOG LOSS ############
# Entrainement de l'algorithme sur les
# 80% des données d'entraînement (fichier train.csv).
nb.fit(train[discrete], train['Numcatego'])

# Prédiction des probabilités pour chaque catégorie de crime,
# sur chaque entrée de crime
probabilities = np.array(nb.predict_proba(perf[discrete]))

# Création d'un DataFrame contenant les probabilités prédites
proba_to_dataframe_perf = pd.DataFrame(probabilities, columns=encoder.classes_)

# calcul du log loss de la solution, unitée utilisée par Kaggle pour mesurer la
# performance du résultat soumis (grâce à la fonction log_loss):
print("LOG LOSS MULTINOMIAL NB : ", log_loss(perf['Numcatego'], probabilities))


########### PARTIE RESULTAT ############
nb = MultinomialNB()
nb.fit(new_train[discrete], new_train['Numcatego'])
probabilities = nb.predict_proba(new_test[discrete])

############ PRECISION SCORE DE PREDICTION DE CHAQUE CATEGORIE ############
###########################################################################
dummiePerf = pd.get_dummies(perf.Numcatego)
dummiePerfArray = np.array(dummiePerf)
print("DUMMIE PERF ARRAY : ", dummiePerfArray)
dummiePerfLabel = pd.DataFrame(dummiePerfArray, columns=encoder.classes_)
print("DUMMIE PERF LABEL : ", dummiePerfLabel.head())


############ ARSON ############
proba = np.array(proba_to_dataframe_perf['ARSON'])
vrai = np.array(dummiePerfLabel['ARSON'])
ARSON_PRECISION_SCORE = average_precision_score(vrai, proba)
print("ARSON PRECISION SCORE : ", ARSON_PRECISION_SCORE)

############ ASSAULT ############
proba = np.array(proba_to_dataframe_perf['ASSAULT'])
vrai = np.array(dummiePerfLabel['ASSAULT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("ASSAULT PRECISION SCORE : ", PRECISION_SCORE)

############ BAD CHECKS ############
proba = np.array(proba_to_dataframe_perf['BAD CHECKS'])
vrai = np.array(dummiePerfLabel['BAD CHECKS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("BAD CHECKS PRECISION SCORE : ", PRECISION_SCORE)

############ BRIBERY ############
proba = np.array(proba_to_dataframe_perf['BRIBERY'])
vrai = np.array(dummiePerfLabel['BRIBERY'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("BRIBERY PRECISION SCORE : ", PRECISION_SCORE)

############ BURGLARY ############
proba = np.array(proba_to_dataframe_perf['BURGLARY'])
vrai = np.array(dummiePerfLabel['BURGLARY'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("BURGLARY PRECISION SCORE : ", PRECISION_SCORE)

############ DISORDERLY CONDUCT ############
proba = np.array(proba_to_dataframe_perf['DISORDERLY CONDUCT'])
vrai = np.array(dummiePerfLabel['DISORDERLY CONDUCT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("DISORDERLY CONDUCT PRECISION SCORE : ", PRECISION_SCORE)

############ DRIVING UNDER THE INFLUENCE ############
proba = np.array(proba_to_dataframe_perf['DRIVING UNDER THE INFLUENCE'])
vrai = np.array(dummiePerfLabel['DRIVING UNDER THE INFLUENCE'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("DRIVING UNDER THE INFLUENCE PRECISION SCORE : ", PRECISION_SCORE)

############ DRUG/NARCOTIC ############
proba = np.array(proba_to_dataframe_perf['DRUG/NARCOTIC'])
vrai = np.array(dummiePerfLabel['DRUG/NARCOTIC'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("DRUG/NARCOTIC PRECISION SCORE : ", PRECISION_SCORE)

############ DRUNKENNESS ############
proba = np.array(proba_to_dataframe_perf['DRUNKENNESS'])
vrai = np.array(dummiePerfLabel['DRUNKENNESS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("DRUNKENNESS PRECISION SCORE : ", PRECISION_SCORE)

############ EMBEZZLEMENT ############
proba = np.array(proba_to_dataframe_perf['EMBEZZLEMENT'])
vrai = np.array(dummiePerfLabel['EMBEZZLEMENT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("EMBEZZLEMENT PRECISION SCORE : ", PRECISION_SCORE)

############ EXTORTION ############
proba = np.array(proba_to_dataframe_perf['EXTORTION'])
vrai = np.array(dummiePerfLabel['EXTORTION'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("EXTORTION PRECISION SCORE : ", PRECISION_SCORE)

############ FAMILY OFFENSES ############
proba = np.array(proba_to_dataframe_perf['FAMILY OFFENSES'])
vrai = np.array(dummiePerfLabel['FAMILY OFFENSES'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("FAMILY OFFENSES PRECISION SCORE : ", PRECISION_SCORE)

############ FORGERY/COUNTERFEITING ############
proba = np.array(proba_to_dataframe_perf['FORGERY/COUNTERFEITING'])
vrai = np.array(dummiePerfLabel['FORGERY/COUNTERFEITING'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("FORGERY/COUNTERFEITING PRECISION SCORE : ", PRECISION_SCORE)

############ FRAUD ############
proba = np.array(proba_to_dataframe_perf['FRAUD'])
vrai = np.array(dummiePerfLabel['FRAUD'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("FRAUD PRECISION SCORE : ", PRECISION_SCORE)

############ GAMBLING ############
proba = np.array(proba_to_dataframe_perf['GAMBLING'])
vrai = np.array(dummiePerfLabel['GAMBLING'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("GAMBLING PRECISION SCORE : ", PRECISION_SCORE)

############ KIDNAPPING ############
proba = np.array(proba_to_dataframe_perf['KIDNAPPING'])
vrai = np.array(dummiePerfLabel['KIDNAPPING'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("KIDNAPPING PRECISION SCORE : ", PRECISION_SCORE)

############ LARCENY/THEFT ############
proba = np.array(proba_to_dataframe_perf['LARCENY/THEFT'])
vrai = np.array(dummiePerfLabel['LARCENY/THEFT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("LARCENY/THEFT PRECISION SCORE : ", PRECISION_SCORE)

############ LIQUOR LAWS ############
proba = np.array(proba_to_dataframe_perf['LIQUOR LAWS'])
vrai = np.array(dummiePerfLabel['LIQUOR LAWS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("LIQUOR LAWS PRECISION SCORE : ", PRECISION_SCORE)

############ LOITERING ############
proba = np.array(proba_to_dataframe_perf['LOITERING'])
vrai = np.array(dummiePerfLabel['LOITERING'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("LOITERING PRECISION SCORE : ", PRECISION_SCORE)

############ MISSING PERSON ############
proba = np.array(proba_to_dataframe_perf['MISSING PERSON'])
vrai = np.array(dummiePerfLabel['MISSING PERSON'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("MISSING PERSON PRECISION SCORE : ", PRECISION_SCORE)

############ NON-CRIMINAL ############
proba = np.array(proba_to_dataframe_perf['NON-CRIMINAL'])
vrai = np.array(dummiePerfLabel['NON-CRIMINAL'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("NON-CRIMINAL PRECISION SCORE : ", PRECISION_SCORE)

############ OTHER OFFENSES ############
proba = np.array(proba_to_dataframe_perf['OTHER OFFENSES'])
vrai = np.array(dummiePerfLabel['OTHER OFFENSES'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("OTHER OFFENSES PRECISION SCORE : ", PRECISION_SCORE)

############ PORNOGRAPHY/OBSCENE MAT ############
proba = np.array(proba_to_dataframe_perf['PORNOGRAPHY/OBSCENE MAT'])
vrai = np.array(dummiePerfLabel['PORNOGRAPHY/OBSCENE MAT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("PORNOGRAPHY/OBSCENE MAT PRECISION SCORE : ", PRECISION_SCORE)

############ PROSTITUTION ############
proba = np.array(proba_to_dataframe_perf['PROSTITUTION'])
vrai = np.array(dummiePerfLabel['PROSTITUTION'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("PROSTITUTION PRECISION SCORE : ", PRECISION_SCORE)

############ RECOVERED VEHICLE ############
proba = np.array(proba_to_dataframe_perf['RECOVERED VEHICLE'])
vrai = np.array(dummiePerfLabel['RECOVERED VEHICLE'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("RECOVERED VEHICLE PRECISION SCORE : ", PRECISION_SCORE)

############ RUNAWAY ############
proba = np.array(proba_to_dataframe_perf['RUNAWAY'])
vrai = np.array(dummiePerfLabel['RUNAWAY'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("RUNAWAY PRECISION SCORE : ", PRECISION_SCORE)

############ SECONDARY CODES ############
proba = np.array(proba_to_dataframe_perf['SECONDARY CODES'])
vrai = np.array(dummiePerfLabel['SECONDARY CODES'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("SECONDARY CODES PRECISION SCORE : ", PRECISION_SCORE)

############ SEX OFFENSES FORCIBLE ############
proba = np.array(proba_to_dataframe_perf['SEX OFFENSES FORCIBLE'])
vrai = np.array(dummiePerfLabel['SEX OFFENSES FORCIBLE'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("SEX OFFENSES FORCIBLE PRECISION SCORE : ", PRECISION_SCORE)

############ SEX OFFENSES NON FORCIBLE ############
proba = np.array(proba_to_dataframe_perf['SEX OFFENSES NON FORCIBLE'])
vrai = np.array(dummiePerfLabel['SEX OFFENSES NON FORCIBLE'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("SEX OFFENSES NON FORCIBLE PRECISION SCORE : ", PRECISION_SCORE)

############ STOLEN PROPERTY ############
proba = np.array(proba_to_dataframe_perf['STOLEN PROPERTY'])
vrai = np.array(dummiePerfLabel['STOLEN PROPERTY'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("STOLEN PROPERTY PRECISION SCORE : ", PRECISION_SCORE)

############ SUICIDE ############
proba = np.array(proba_to_dataframe_perf['SUICIDE'])
vrai = np.array(dummiePerfLabel['SUICIDE'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("SUICIDE PRECISION SCORE : ", PRECISION_SCORE)

############ SUSPICIOUS OCC ############
proba = np.array(proba_to_dataframe_perf['SUSPICIOUS OCC'])
vrai = np.array(dummiePerfLabel['SUSPICIOUS OCC'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("SUSPICIOUS OCC PRECISION SCORE : ", PRECISION_SCORE)

############ TRESPASS ############
proba = np.array(proba_to_dataframe_perf['TRESPASS'])
vrai = np.array(dummiePerfLabel['TRESPASS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("TRESPASS PRECISION SCORE : ", PRECISION_SCORE)

############ VANDALISM ############
proba = np.array(proba_to_dataframe_perf['VANDALISM'])
vrai = np.array(dummiePerfLabel['VANDALISM'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("VANDALISM PRECISION SCORE : ", PRECISION_SCORE)

############ VEHICLE THEFT ############
proba = np.array(proba_to_dataframe_perf['VEHICLE THEFT'])
vrai = np.array(dummiePerfLabel['VEHICLE THEFT'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("VEHICLE THEFT PRECISION SCORE : ", PRECISION_SCORE)

############ WARRANTS ############
proba = np.array(proba_to_dataframe_perf['WARRANTS'])
vrai = np.array(dummiePerfLabel['WARRANTS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("WARRANTS PRECISION SCORE : ", PRECISION_SCORE)

############ WEAPON LAWS ############
proba = np.array(proba_to_dataframe_perf['WEAPON LAWS'])
vrai = np.array(dummiePerfLabel['WEAPON LAWS'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("WEAPON LAWS PRECISION SCORE : ", PRECISION_SCORE)

############ ROBBERY ############
proba = np.array(proba_to_dataframe_perf['ROBBERY'])
vrai = np.array(dummiePerfLabel['ROBBERY'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("ROBBERY PRECISION SCORE : ", PRECISION_SCORE)

############ TREA ############
proba = np.array(proba_to_dataframe_perf['TREA'])
vrai = np.array(dummiePerfLabel['TREA'])
PRECISION_SCORE = average_precision_score(vrai, proba)
print("TREA PRECISION SCORE : ", PRECISION_SCORE)

# GENERATION DU FICHIER CSV DE SOUMISSION KAGGLE
soumission = pd.DataFrame(probabilities, columns=encoder.classes_)
soumission.to_csv('Result.csv', index=True, index_label='Id')