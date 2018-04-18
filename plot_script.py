import matplotlib.pyplot as plt

plt.plot(error_train, 'b-', label='train error')
plt.plot(error_test, 'k-', label='test error')
plt.xlabel('epoch')
plt.ylabel('error')
plt.title('Train and Test Error')
plt.legend(loc='best')
plt.show()