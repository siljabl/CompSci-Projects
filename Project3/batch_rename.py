import os
path = 'C:/Users/franscho/Documents/CompSci-Projets/Project3/data/amptoph/no_split/amp'
files = os.listdir(path)

raise Exception('CAUTION - files will be renamed. Comment out this exception if you want to continue.')

i = 1 
for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(i).zfill(4), '.png'])))
    i = i+1