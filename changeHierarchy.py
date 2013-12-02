import sys

inputFile = sys.argv[1]
newFile = inputFile.split('.')[0] + '.new.hierarchy'

f = open(inputFile)
fnew = open(newFile,'w')
for line in f:
	split = line.split(' ')
	
	newLine = split[0] + ' ' + ' '.join(split[2:])

	fnew.write(newLine)


