import sys

sum = 0
n = 0

for line in sys.stdin:
  sum += float(line)
  n += 1
  
print(round(sum/n, 3))
