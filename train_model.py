import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
import joblib

print("Memulai proses training model final...")

# Memuat data dari CSV
df = pd.read_csv('BrainTumor.csv')
x = df.drop(columns=['Image', 'Class'])
y = df['Class']

# Membuat dan melatih scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("Data berhasil dimuat dan di-scale.")

# Inisialisasi model-model dasar sesuai parameter di proposal
knn_base = KNeighborsClassifier(n_neighbors=11, metric='manhattan')
alr_base = LogisticRegression(C=0.000082)
svm_base = SVC(kernel='poly', degree=3, probability=True)

estimators = [('KNN', knn_base), ('ALR', alr_base), ('SVM', svm_base)]

# Inisialisasi dan melatih model Ensemble Stacking
final_model = StackingClassifier(estimators=estimators)
final_model.fit(x_scaled, y)
print("Model StackingClassifier selesai dilatih.")

# Menyimpan model dan scaler ke dalam file
joblib.dump(final_model, 'brain_tumor_stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model dan Scaler telah disimpan.")

# --- PENAMBAHAN BARU DIMULAI DI SINI ---
# Menghitung nilai rata-rata dari setiap kolom fitur
default_values = x.mean().to_dict()

# Menyimpan nilai rata-rata ke dalam file pkl baru
joblib.dump(default_values, 'default_values.pkl')
print("Nilai default (rata-rata) telah dihitung dan disimpan sebagai 'default_values.pkl'.")
# --- AKHIR PENAMBAHAN ---
