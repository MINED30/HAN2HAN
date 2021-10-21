import glob
character_list = [f.split('/')[-1] for f in glob.glob('/content/이승우/*.png')]
# character_list = [f+'.jpg' for f in '나랏말싸미듕귁에달아']
character_list