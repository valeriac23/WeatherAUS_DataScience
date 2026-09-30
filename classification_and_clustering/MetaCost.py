import pandas as pd
import numpy as np
from sklearn.base import clone


class MetaCost(object):
    """
    Implementazione del meta-algoritmo MetaCost (Domingos, 1999), che rende un
    classificatore sensibile ai costi d'errore tramite rietichettatura del training
    basata su un ensemble bootstrap.

    Convenzione della matrice dei costi: C[i, j] = costo di predire la classe i
    quando la classe vera è j. Con C = [[0, 3], [1, 0]] un falso negativo costa 3
    e un falso allarme costa 1, e un esempio viene rietichettato come positivo
    quando la sua probabilità stimata supera C10 / (C10 + C01) = 0,25.
    """

    def __init__(self, S, L, C, m=50, n=1, p=True, q=True, random_state=None):
        """
        :param S: training set (DataFrame con feature e colonna target)
        :param L: modello di classificazione di base (stimatore scikit-learn)
        :param C: matrice dei costi, C[i, j] = costo di predire i quando la verità è j
        :param m: numero di campioni bootstrap (default 50)
        :param n: dimensione di ogni campione bootstrap come frazione di len(S) (default 1)
        :param p: True se il modello L espone predict_proba
        :param q: True per stimare P(j|x) con tutti i modelli dell'ensemble;
                  False per usare solo i modelli che non hanno visto l'esempio (out-of-bag)
        :param random_state: seed per la riproducibilità dei campioni bootstrap
        """
        if not isinstance(S, pd.DataFrame):
            raise ValueError("S must be a DataFrame object")
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C deve essere una matrice quadrata num_class x num_class")
        self.S = S.reset_index(drop=True).copy()   # copia: non modifica il DataFrame del chiamante
        self.L = L
        self.C = C
        self.m = m
        self.n = int(round(len(S) * n))
        self.p = p
        self.q = q
        self.rng = np.random.default_rng(random_state)

    def fit(self, flag, num_class):
        """
        Calcola le nuove etichette a costo minimo e addestra il modello finale.

        :param flag: nome della colonna target in S (classi codificate come 0, ..., num_class-1)
        :param num_class: numero totale di classi distinte
        """
        if self.C.shape[0] != num_class:
            raise ValueError("La dimensione di C non corrisponde a num_class")

        col = [c for c in self.S.columns if c != flag]
        X_full = self.S[col].values
        y_full = self.S[flag].values.astype(int)
        N = len(self.S)

        P_sum = np.zeros((N, num_class))
        counts = np.zeros((N, 1))
        P_all_sum = np.zeros((N, num_class))       # somma su tutti i modelli (ripiego OOB)

        # 1. Ensemble bootstrap e stima delle probabilità di classe
        for _ in range(self.m):
            sample_idx = self.rng.choice(N, size=self.n, replace=True)
            model = clone(self.L)
            model.fit(X_full[sample_idx], y_full[sample_idx])

            probas = np.zeros((N, num_class))
            if self.p:
                # Le colonne di predict_proba seguono model.classes_: le si riporta nella
                # posizione della classe corrispondente, cosi' il calcolo resta corretto
                # anche se un campione bootstrap non contenesse tutte le classi.
                probas[:, model.classes_.astype(int)] = model.predict_proba(X_full)
            else:
                probas[np.arange(N), model.predict(X_full).astype(int)] = 1.0

            P_all_sum += probas
            if self.q:
                P_sum += probas
                counts += 1
            else:
                is_oob = np.ones(N, dtype=bool)
                is_oob[sample_idx] = False
                P_sum[is_oob] += probas[is_oob]
                counts[is_oob] += 1

        # 2. Probabilità media P(j|x). Con q=False, gli esempi mai rimasti fuori dal
        #    bootstrap (circa lo 0,632^m dei casi) non avrebbero stima: si usa la media
        #    su tutti i modelli invece di lasciare P = 0, che li etichetterebbe come classe 0.
        P_avg = P_all_sum / self.m
        mask = counts[:, 0] > 0
        P_avg[mask] = P_sum[mask] / counts[mask]

        # 3. Rietichettatura: classe che minimizza il costo atteso sum_j P(j|x) C[i, j]
        expected_risks = P_avg.dot(self.C.T)
        self.new_labels_ = np.argmin(expected_risks, axis=1)
        self.n_relabeled_ = int((self.new_labels_ != y_full).sum())

        # 4. Modello finale addestrato sul training originale con le nuove etichette
        model_new = clone(self.L)
        model_new.fit(X_full, self.new_labels_)
        return model_new