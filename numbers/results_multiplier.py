import subprocess
import sys


processes = []
for i in range(int(sys.argv[1])):
    p = subprocess.Popen('python3 difl.py MNIST_Data/normal/ MNIST_Data/inverted/ 3', stdout = subprocess.PIPE, shell=True)
    processes.append(p)

results = []
for process in processes:
    results.append(process.communicate()[0].decode().split("\n")[-2])

with open("results.txt", "w") as f:
    for i in range(len(results)):
        f.write(results[i])
        f.write("\n")

print("\nDone!")
