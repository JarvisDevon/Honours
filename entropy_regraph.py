import numpy as np
import matplotlib.pyplot as plt

f = open('Run7/log.txt')
file_text = f.read()
split = file_text.split('\n')
split = split[4:]
split = split[:-1]
hold = np.zeros(22)
for elem in split:
	split_char = elem.split(',')
	nump_arr = np.array(split_char)
	hold = np.vstack([hold, nump_arr])

hold1 = hold[2:,0:4]	# Second index determines whats graphed
hold1 = hold1.astype(np.float64)
hold1 = np.round(hold1, decimals=5)
hold2 = hold[2:,-2]
hold2 = hold2.astype(np.float64)
hold2 = np.round(hold2, decimals=5)
hold3 = hold[2:,14]
hold3 = hold3.astype(np.complex128).astype(np.float64)
hold3 = np.round(hold3, decimals=5)
#print(hold3)
vert_line = np.argmax(hold3)
like_int = np.argmin(np.abs(hold1[0:400, 0] - hold1[0:400, 3]))
error_min = np.argmin(hold2)
print(vert_line)
plt.plot(hold3, color='blue', linewidth=2.3)
#plt.fill_between(np.arange(144,203,1), hold3[144:203], color = "green")
plt.axvline(error_min,color='red', linewidth=2.3)
plt.axvline(like_int, color='orange', linewidth=2.3)
plt.axvline(vert_line,color='lime', linewidth=2.3)
plt.axhline(0,color='black', linewidth=2.0)
plt.legend(['Model Entropy on Principal Sub-manifold','Test Error Minimum', 'Equality of Likelihoods','Maximum Entropy'], fontsize=12)
plt.xlabel('Number of Training Steps \n (c)', fontsize=14)
plt.ylabel('Entropy', fontsize=14)
plt.tight_layout()
plt.savefig('model_entropy.svg')
