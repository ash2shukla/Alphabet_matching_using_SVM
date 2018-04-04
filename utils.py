import pandas as pd
from numpy import apply_along_axis
from sklearn import svm
from pickle import dump,load
import matplotlib.pyplot as plt
from string import ascii_uppercase

def get_data(_path="./data/Alpha.csv.dat"):
	names = ['lettr','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2br','xy2br','x-ege','xegvy','y-ege','yegvx']	
	df = pd.read_csv(_path, names=names)
	# Convert to numpy array as dat_arr
	dat_arr = df.as_matrix()
	# data_rows
	rows = dat_arr.shape[0]
	output = dat_arr[:,0].reshape(rows,-1)
	output = apply_along_axis(lambda row : ord(row[0])-ord('A'),1,output)
	features = dat_arr[:,1:]
	return output,features

def get_train_test(perc=75):
	output,features = get_data()
	total_rows = output.shape[0]
	train_rows = int(total_rows*perc/100)

	train_input = features[:train_rows,:]
	train_output = output[:train_rows]

	test_input = features[train_rows:,:]
	test_output = output[train_rows:]

	return train_input,train_output,test_input,test_output

def train():
	clf = svm.SVC()
	train_input,train_output,test_input,test_output = get_train_test()
	clf.fit(train_input,train_output)
	with open('./trained_model/AlphaClassifier.pkl', 'wb') as svm_alpha:
		dump(clf, svm_alpha)

def test():
	model_path = './trained_model/AlphaClassifier.pkl'
	model = load(open(model_path,'rb'))
	print('Pretrained model loaded from',model_path)
	_,_,test_input,test_output = get_train_test()
	test_output_map = {}
	for i in ascii_uppercase:
		test_output_map[i] ={}
	total=len(test_output)
	#{"actual": {"predicted":times}}
	for i,j in enumerate(test_input):
		print('Testing in progress =',str(((i+1)*100)/total),'%',end='\r')
		predicted = chr(model.predict(j.reshape(1,-1))[0]+ord('A'))
		try:
			test_output_map[chr(test_output[i]+ord('A'))][predicted]+=1
		except KeyError:
			test_output_map[chr(test_output[i]+ord('A'))][predicted]=1
	return test_output_map
def getAccuracy():
	print('Calculating Accuracy')
	model_path = './trained_model/AlphaClassifier.pkl'
	model = load(open(model_path,'rb'))
	_,_,test_input,test_output = get_train_test()
	print('The accuracy of model is',model.score(test_input, test_output)*100)
	
def print_mismatchMatrix(test_output_map={}):
	print('  ',end='')
	print(''.join([i.ljust(4) for i in ascii_uppercase]))
	for i in ascii_uppercase:
		temp_dict = test_output_map[i]
		temp_arr = []
		for j in ascii_uppercase:
			try:
				temp_arr.append(temp_dict[j])
			except KeyError:
				temp_arr.append(0)
		print(i,''.join([str(k).ljust(4) for k in temp_arr]))

def plot_dict(D={},title='Example_Title',ylbl='',xlbl='',do_save=False):
	plt.title(title)
	plt.ylabel(ylbl)
	plt.xlabel(xlbl)
	plt.bar(range(len(D)), list(D.values()), align='center')
	plt.xticks(range(len(D)), list(D.keys()))
	plt.savefig(title+'.png') if do_save else plt.show()
	plt.close()

def plot_errors(test_output_map={}):
	errors = {}
	for i in ascii_uppercase:
		total_tests = sum(test_output_map[i].values())
		errors[i] =100-(test_output_map[i][i]/total_tests)*100
	plot_dict(errors,'Errors for each alphabet','Error Percentage','Alphabets',False)

def plot_mismatch(test_output_map={'A':{}},alpha = 'A'):
	test_output_map[alpha].pop(alpha)
	plot_dict(test_output_map[alpha],'Mismatched for alphabet '+alpha,'times mismatched','Alphabets',False)
