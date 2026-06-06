from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

best_model = models.load_model(best_model_path)

true_labels = val_gen.classes

pred_probs = best_model.predict(val_gen)
pred_labels = np.argmax(pred_probs, axis=1)

class_names = list(val_gen.class_indices.keys())
print("Class Names:", class_names)

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n")
print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=class_names
    )
)