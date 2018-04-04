from string import ascii_uppercase
from utils import plot_errors, plot_mismatch, print_mismatchMatrix,test, getAccuracy

test_map = test()
print_mismatchMatrix(test_map)
getAccuracy()
#Uncomment following lines to show plots of errors and mismatches on complete
#plot_errors(test_map)
#[plot_mismatch(test_map,i) for i in ascii_uppercase]
