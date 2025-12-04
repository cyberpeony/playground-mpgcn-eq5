import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["Transit", "Social_People", "Play_Object_Normal"]

preds = np.load("../data/preds_test.npy")
labels = np.load("../data/labels_test.npy")

print("Predicciones cargadas:", preds.shape)
print("Labels cargados:", labels.shape)

#accuracy global
test_acc = accuracy_score(labels, preds)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

#accuracy por clase
print("\n Accuracy por clase:")
for idx, class_name in enumerate(CLASS_NAMES):
    mask = labels == idx
    if mask.sum() == 0:
        print(f"  {class_name}: (sin muestras en test)")
        continue
    class_acc = (preds[mask] == labels[mask]).mean()
    print(f"  {class_name}: {class_acc*100:.2f}%")

#classification report
print("\n Classification Report:")
print(
    classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    )
)
#matriz de confusión 
cm = confusion_matrix(labels, preds)
print("\n Confusion Matrix:")
print(cm)

#plot matriz de confusión
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Matriz de Confusión – Test Set", fontsize=14)
plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
plt.yticks(tick_marks, CLASS_NAMES)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=200)
print("\n Imagen guardada como: confusion_matrix_test.png")
plt.close()

# precision, recall, f1 por clase
prec, rec, f1, support = precision_recall_fscore_support(
    labels, preds, labels=np.arange(len(CLASS_NAMES))
)

# gráfica f1 por clase
plt.figure(figsize=(6, 4))
x = np.arange(len(CLASS_NAMES))
plt.bar(x, f1)
plt.xticks(x, CLASS_NAMES, rotation=30, ha="right")
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("F1-score por clase (Test)")
plt.tight_layout()
plt.savefig("f1_per_class.png", dpi=200)
print(" Imagen guardada como: f1_per_class.png")
plt.close()

# gráfica precision / recall por clase
width = 0.35
plt.figure(figsize=(6, 4))
x = np.arange(len(CLASS_NAMES))
plt.bar(x - width/2, prec, width, label="Precision")
plt.bar(x + width/2, rec,  width, label="Recall")
plt.xticks(x, CLASS_NAMES, rotation=30, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Precision y Recall por clase (Test)")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_per_class.png", dpi=200)
print(" Imagen guardada como: precision_recall_per_class.png")
plt.close()