import numpy as np


A = np.array([[.5, .4, .2],
              [.1, .4, .3],
              [.4, .2, .5]])


print(A)

eig = np.linalg.eig(A)[0]
print(eig)

frob=max(eig)
print("число Фробеніуса: ", frob)

print("Лівий вектор Фробеніуса: ",np.linalg.eig(A)[1][:,1])

AT=A.T
print("транспонована матриця:")
print(AT)

print("Правий вектор Фробеніуса: ")

eigr = np.linalg.eig(A.T)[0]
# print((eigr))
print(np.linalg.eig(A.T)[1][:,1])


print("Продуктивність матриці:")
print(f"{frob} <=1 - {frob <= 1}")


# print("Матриця повних витрат: ")
AA = np.array([[.6, -0.1, -0.5],
              [-0.9, .4, -0.3],
              [-0.5, -0.7, 0.8]])

B = np.linalg.inv(AA)

# print(B)

A1 = B - np.eye(3)
i = 1
last_max = np.amax(A1)
while 1:
  i += 1
  A1 = A1 - np.linalg.matrix_power(A, i)
  if last_max - np.amax(A1) < 0.001:
    last_max = np.amax(A1)
    break
  if last_max<-1000:
    break
  #print(last_max)
  last_max = np.amax(A1)

print("збіжність матриці А :")
print(f"{last_max} < 0.01 - {(abs(last_max) < 0.01)}")

print(np.array([.4, .3, .6]) @ B)

