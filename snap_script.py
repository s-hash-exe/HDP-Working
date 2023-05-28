import sys
sys.path.insert(0, r'E:\Work\HDP-Working\HDP')
from HDP import RefineContourMyo
print(sys.path)
def main_func():
    epicardium = RefineContourMyo.mainFunc()
    print("Size of Received Epicardium : {}".format(len(epicardium)))

if __name__ == '__main__':
    main_func()
