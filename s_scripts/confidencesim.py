import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# number of simulations
N = 50

# load data
df = pd.read_csv('../data/train.csv', '|')

# create result file
result_file = '../data/confidence_sim.csv'
with open(result_file, 'w') as f:
    f.write('0_confidence,1_confidence,predicted_class,real_class')

for i in range(N):
    X, y = df.drop('fraud', axis=1), df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ro = RandomOverSampler()
    X_res, y_res = ro.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(bootstrap=True,
                                  max_depth=80,
                                  max_features='sqrt',
                                  min_samples_leaf=2,
                                  min_samples_split=5,
                                  n_estimators=1000)

    clf.fit(X_res, y_res)

    prob = list(clf.predict_proba(X_test))
    prob = [list(p) for p in prob]

    for p in prob:
        if p[0] > p[1]:
            p.append(0)
        else:
            p.append(1)

    for i in range(len(y_test)):
        prob[i].append(list(y_test)[i])

    with open(result_file, 'a') as f:
        for p in prob:
            if p[-1] != p[-2]:
                f.write('{},{},{},{}\n'.format(*p))