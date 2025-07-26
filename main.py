x = [7, 9, 11, 15, 17, 21, 24]
y = [39, 56, 63, 73, 85, 93, 100]

meanofx = sum(x) / len(x)
meanofy = sum(y) / len(y)

upxarr = []
upyarr = []

for i in range(len(x)):
    upxarr.append(meanofx - x[i])
    upyarr.append(meanofy - y[i])

up = sum(upxarr)
down = sum(upyarr)

up = sum((meanofx - x[i]) * (meanofy - y[i]) for i in range(len(x)))
down = sum((meanofx - x[i]) ** 2 for i in range(len(x)))

print("up:", up)
print("down:", down)
slope = up / down

intercept = meanofy - slope * meanofx

print("Slope:", round(slope, 2))
print("Intercept:", round(intercept, 2))

lr = 0.02

ycap = [slope * xi + intercept for xi in x]
print("Predicted y values:", ycap)


loss = sum((yi - ycap[i]) ** 2 for i, yi in enumerate(y)) / len(y)
print("Loss:", loss)

w = slope
b = intercept
temp = 0

for i in range(len(x)):
    for j in range(len(y)):
        temp += (w * i + b - j)* i

weight = (2 /len(x)) * temp

temp = 0

for i in range(len(x)):
    for j in range (len(y)):
        temp += (w * i + b - j)

bias = (2 / len(x)) * temp


print("\nWeight: ",weight)
print("Bias: ",bias)



newweight = w - lr * weight
newbias = b -lr * bias


print("\nnew weight: ", newweight)
print("new bias: ", newbias)


mae = sum(abs(yi - ycap[i]) for i, yi in enumerate(y)) / len(y)
print("\nMean Absolute Error (MAE):", round(mae, 2))
mse = sum((yi - ycap[i]) ** 2 for i, yi in enumerate(y)) / len(y)
print("\nMean Squared Error (MSE):", round(mse, 2))
rmse = (mse) ** 0.5
print("\nRoot Mean Squared Error (RMSE):", round(rmse, 2))