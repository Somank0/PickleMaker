import pickle

file = open("trueE_target.pickle", "rb")
data = pickle.load(file)
weights = [int(i > 105) * 100 + 1 for i in data]
with open("trueE_target.pickle", "wb") as outpickle:
    pickle.dump(weights, outpickle)
