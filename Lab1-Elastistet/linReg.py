import numpy as np
from sklearn.linear_model import LinearRegression

# x: mass[kg], y: height [mm]
x = np.array([[0.0000000e+00], [5.0000000e-01], [1.0000000e+00], [1.5000000e+00], [2.0000000e+00], [2.5000000e+00]])
y = np.array([8.81, 8.09, 7.27,6.59, 5.84, 5.07])

model = LinearRegression()
model.fit(x, y)
A = model.coef_[0]
print(f"Stigningstall {A:.3f}")


#Opg 5:Usikkerhet delta A
#https://www.econometrics-with-r.org/5.2-cifrc.html
data = np.column_stack((x, y))
print(data)
std_err = np.std(data, ddof=1) / np.sqrt(6)
print(f"std_err: {std_err}")

#Difference predict vs data
diff = y - model.predict(x)
err = np.sum(diff)/np.count_nonzero(diff)
print(f"Avg diff predicted vs data: {err}")

#Opg 7: ∆f =1/T

