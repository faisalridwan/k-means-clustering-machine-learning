
# Faisal Ridwan (1301174010)
# Ilham Rizki Julianto (1301170293)
# IF-41-06

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
data = pd.read_csv('DataSet.csv')
print("Input Data and Shape")
print(data.shape)
data

# Plot dataset ke Grafik
f1 = data['x'].values
f2 = data['y'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, color='red')
plt.show


# Function Hitung Jarak (eculidian)
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Jumlah Cluster
k = 3
# k = int(input("Masukkan Jumlah Cluster: "))

# Inisialisasi Koordinat Cantroid Random sesuai dataset (x,y)
C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)


# Print Initial Centorids
print("Initial Centroids: ")
print("Koordinat X: ")
print(C_x)
print("Koordinat Y: ")
print(C_y)

# Plot Centorid ke Grafik
plt.scatter(f1, f2, c='red')
plt.scatter(C_x, C_y, c='black')
plt.close()

# Simpan Nilai Centroid lama
C_old = np.zeros(C.shape)

# Label Cluster(0, 1, 2)
clusters = np.zeros(len(X))

# Function Error. - Jarak antara centroid baru dan centroid lama
error = dist(C, C_old, None)

# Loop akan berjalan sampai error menjadi nol (tidak ada pergerakan antara centroid lama dan yg baru)
while error != 0:
    # Menetapkan setiap nilai ke cluster terdekat
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    # Menyimpan nilai centroid lama
    C_old = deepcopy(C)

    # Menemukan centroid baru dengan mengambil nilai rata-rata
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

# Plotting Hasil ke Grafik
colors = 10*["r", "g", "b"]
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j]
                       for j in range(len(X))
                       if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], c=colors[i])
    print(points[:, 0])
    print(points[:, 1])
    print(C[i, 0], C[i, 1])
    print("==")

ax.scatter(C[:, 0], C[:, 1], c='black', s=80)
plt.show()
