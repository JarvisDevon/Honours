import numpy as np
import matplotlib.pyplot as plt

f = open('Run1*/hess.txt')
file_text = f.read()
split = file_text.split(']\n[')
hold = np.zeros(870)
for elem in split:
	time_step = np.array([])
	split_char = elem.split(' ')
	if split_char[0] == '[':
		split_char = split_char[1:]
	if split_char[-1] == ']':
		split_char = split_char[:-1]
	nump_arr = np.array(split_char)
	for item in nump_arr[:-1]:
		if not item == '':
			if item[-2:] == '\n':
				add_str = item[:-2]
			else:
				add_str = item
			print(add_str)
			time_step = np.append(time_step, complex(add_str))
	hold = np.vstack([hold, time_step])

plt.plot(hold[1:100, 0], linewidth=2.3)
plt.axhline(0,color='black', linewidth=2.0)
plt.axvline(4,color='red', linewidth=2.3)
plt.legend(['Principal Curvature Number 1'], fontsize=12)
#plt.title('6th Principal Curvature', fontsize=16)
plt.xlabel('Number of Training Steps \n (a)', fontsize=14)
plt.ylabel('Curvature', fontsize=14)
plt.tight_layout()
plt.savefig('prince_curv1.svg')
plt.close()

plt.plot(hold[1:100, 2], linewidth=2.3)
plt.axhline(0,color='black', linewidth=2.0)
plt.axvline(4,color='red', linewidth=2.3)
plt.legend(['Principal Curvature Number 3'], fontsize=12)
#plt.title('6th Principal Curvature', fontsize=16)
plt.xlabel('Number of Training Steps \n (b)', fontsize=14)
plt.ylabel('Curvature', fontsize=14)
plt.tight_layout()
plt.savefig('prince_curv3.svg')
plt.close()

plt.plot(hold[1:100, 4], linewidth=2.3)
plt.axhline(0,color='black', linewidth=2.0)
plt.axvline(4,color='red', linewidth=2.3)
plt.legend(['Principal Curvature Number 5'], fontsize=12)
#plt.title('6th Principal Curvature', fontsize=16)
plt.xlabel('Number of Training Steps \n (c)', fontsize=14)
plt.ylabel('Curvature', fontsize=14)
plt.tight_layout()
plt.savefig('prince_curv5.svg')
