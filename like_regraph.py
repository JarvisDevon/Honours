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
vert_line = np.argmin(hold2)
like_int = np.argmin(np.abs(hold1[0:400, 0] - hold1[0:400, 3]))
print(vert_line)
plt.plot(hold1[2:100, 0], linewidth=2.3)
plt.plot(hold1[2:100, 3], color = 'lime', linewidth=2.3)
#plt.plot(hold2[0:200])
plt.axvline(vert_line,color='red', linewidth=2.3)
plt.axvline(like_int,color='orange', linewidth=2.3)
plt.axhline(0,color='black', linewidth=2.0)
plt.legend(['Data Likelhood from Model','Data Likelihood from True Distribution','Test Error Minimum', 'Equality of Likelihoods'], fontsize=12)
plt.xlabel('Number of Training Steps \n (b)', fontsize=12)
plt.ylabel('Likelihood', fontsize=14)
plt.tight_layout()
plt.savefig('like.svg')
