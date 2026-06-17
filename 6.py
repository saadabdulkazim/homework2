import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# البيانات من السؤال الأول
x = np.array([1,3,6])
y = np.array([6,10,16])

# معاملات البداية
theta0 = 0
theta1 = 0

alpha = 0.05
iterations = 50

m = len(x)

theta0_history = []
theta1_history = []
rss_history = []

# Gradient Descent
for i in range(iterations):

    y_pred = theta0 + theta1*x

    error = y_pred - y

    rss = np.sum(error**2)

    theta0_history.append(theta0)
    theta1_history.append(theta1)
    rss_history.append(rss)

    d_theta0 = (2/m)*np.sum(error)
    d_theta1 = (2/m)*np.sum(error*x)

    theta0 -= alpha*d_theta0
    theta1 -= alpha*d_theta1


# إنشاء شبكة theta0 و theta1 لرسم الـ Contour
theta0_vals = np.linspace(-5,10,100)
theta1_vals = np.linspace(0,5,100)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

RSS = np.zeros(T0.shape)

for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):

        y_hat = T0[i,j] + T1[i,j]*x

        RSS[i,j] = np.sum((y-y_hat)**2)


# إنشاء الشكل
fig, ax = plt.subplots(1,2,figsize=(12,5))

def update(frame):

    ax[0].clear()
    ax[1].clear()

    # Contour plot
    ax[0].contour(T0,T1,RSS,levels=20)

    ax[0].plot(theta0_history[:frame+1],
               theta1_history[:frame+1],
               'ro-')

    ax[0].set_xlabel("θ0")
    ax[0].set_ylabel("θ1")
    ax[0].set_title("Contour Plot")

    # Scatter plot + regression line
    ax[1].scatter(x,y,s=80)

    y_line = theta0_history[frame] + theta1_history[frame]*x

    ax[1].plot(x,y_line,'r',linewidth=3)

    ax[1].set_xlim(0,7)
    ax[1].set_ylim(0,20)

    ax[1].set_title("Linear Regression")

    fig.suptitle(
        f"Iteration = {frame}      RSS = {rss_history[frame]:.2f}"
    )


ani = FuncAnimation(
    fig,
    update,
    frames=iterations,
    interval=200
)

plt.show()