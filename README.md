# GABD Practica 2. Gestió de Dades Massives i Optimització de Consultes
## Introducció
En aquesta pràctica treballarem la importació massiva de dades en Oracle utilitzant Python, les gestionarem i avaluarem l’impacte dels índexs en l’execució de consultes. Per la importació de dades s’haurà d’implementar un script de Python que permeti inserir les dades del repositori de la UCI . Aquest repositori conté conjunts de dades de diferents tipus i característiques que s’utilitzen per avaluar mètodes d’aprenentatge computacional. Per la gestió de dades s’haurà d’adaptar el script de Python lliurat amb la pràctica per que sigui capaç de carregar el conjunt de dades, realitzar el experiment que s’especifiqui i guardar els resultats obtinguts a la base de dades. Finalment, amb tot el volum de dades generat, s’avaluarà l’impacte dels índexs en les consultes a fer.

## Materials i Recursos
Per fer el lliurament d’aquest part disposeu dels següents materials i recursos:
- L'enunciat de la [pràctica](Enunciat_Practica_2.pdf).
- Scripts Python de referència: El codi d'aquest repositori 
- Dades d’exemple de la UCI: Iris, Ionosphere, breast-cancer, letter-recognition. Us els podeu baixar del repositori oficial o del CV de l'assignatura

A més, necessitareu els següents programes per a realitzar les pràctiques i monitoritzar els SGBD: SQLDeveloper, Pycharm (IDE per programar en Python). Caldrà que disposeu d’un client SSH per connectar-vos a les màquines de les pràctiques. A més a la màquina main del projecte teniu instal·lat un intèrpret de Python amb les llibreries Oracle Instant Client i els paquets bàsics per executar els vostres scripts. A més teniu instal·lada la comanda tmux , aquesta comanda permet obrir sessions en la terminal i recuperar-la en una nova sessió. És a dir, us permet entrar i sortir de la màquina main per ssh mantenint el que havia en la sessió anterior.

## Contacte
Per resoldre els dubtes i qüestions que pugueu tenir de la pràctica, podeu contactar als professors de l'assignatura:
- Oriol Ramos (oriol.ramos@uab.cat)
