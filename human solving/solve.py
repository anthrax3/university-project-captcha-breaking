from antigate import AntiGate
gate = AntiGate('40207091f8f58a38c3bb136b5a8688fe')

idsFile = 'ids.txt'
solutionsFile = 'solutions.txt'


for i in range(1, 2000):
	captcha_id = gate.send('../samples/'+str(i)+'.png')
	with open(idsFile, 'a') as f:
		f.write(str(i) + '\t' + captcha_id + '\n')

	solved = gate.get(captcha_id)
	solved = solved.replace("\n","")
	solved = solved.replace("\t","")

	with open(solutionsFile, 'a') as f:
		f.write(str(i) + '\t' + solved + '\n')

	

