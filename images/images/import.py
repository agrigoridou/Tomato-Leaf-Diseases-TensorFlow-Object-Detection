import pandas as pd

# Διαβάζουμε το αρχείο CSV με τις ετικέτες
data = pd.read_csv('train_labels.csv')

# Βρίσκουμε τον αριθμό των μοναδικών κλάσεων
num_classes = len(data['class'].unique())

print("Ο αριθμός των μοναδικών κλάσεων είναι:", num_classes)
