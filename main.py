
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pandas as pd

print("\n=== RECOMMENDATION SYSTEM USING COLLABORATIVE FILTERING ===\n")

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print("\nMODEL EVALUATION")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

def get_top_n_recommendations(predictions, n=10):
    from collections import defaultdict

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


top_n = get_top_n_recommendations(predictions, n=5)

print("\n=== SAMPLE RECOMMENDATIONS ===")
for i, (user, items) in enumerate(top_n.items()):
    print(f"\nUser ID: {user}")
    for item, rating in items:
        print(f"  Item: {item}, Predicted Rating: {rating:.2f}")
    if i == 4:
        break


print("\n=== CROSS VALIDATION RESULTS ===")
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
