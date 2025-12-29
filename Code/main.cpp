/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <iomanip>
using namespace std;

constexpr int CLASSES = 4;

struct Metrics {
    long double accuracy;
    long double precision;
    long double recall;
    long double f1;
};

Metrics computeMetrics(const int conf[CLASSES][CLASSES], int classes);

int main()
{
	//VGG16
	int vgg16[CLASSES][CLASSES] = {
	    {274, 25, 0, 1},
	    {4, 284, 8, 10},
	    {0, 0, 405, 0},
	    {1, 1, 0, 298},
	};
	
	Metrics vgg16M = computeMetrics(vgg16, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] VGG16 [===]\n";
	cout << "Accuracy: " << vgg16M.accuracy << ";\n";
	cout << "Precision: " << vgg16M.precision << ";\n";
	cout << "Recall: " << vgg16M.recall << ";\n";
	cout << "F1-Score: " << vgg16M.f1 << ".\n\n";

	//VGG19
	int vgg19[CLASSES][CLASSES] = {
	    {283, 17, 0, 0},
	    {6, 299, 1, 0},
	    {0, 2, 403, 0},
	    {3, 16, 0, 281},
	};
	
	Metrics vgg19M = computeMetrics(vgg19, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] VGG19 [===]\n";
	cout << "Accuracy: " << vgg19M.accuracy << ";\n";
	cout << "Precision: " << vgg19M.precision << ";\n";
	cout << "Recall: " << vgg19M.recall << ";\n";
	cout << "F1-Score: " << vgg19M.f1 << ".\n\n";


	//ResNet18
	int resnet18[CLASSES][CLASSES] ={
	    {288, 12, 0, 0},
	    {0, 303, 3, 0},
	    {0, 0, 405, 0},
	    {0, 2, 0, 298},
	};
	
	Metrics resnet18M = computeMetrics(resnet18, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] ResNet18 [===]\n";
	cout << "Accuracy: " << resnet18M.accuracy << ";\n";
	cout << "Precision: " << resnet18M.precision << ";\n";
	cout << "Recall: " << resnet18M.recall << ";\n";
	cout << "F1-Score: " << resnet18M.f1 << ".\n\n";

	//ResNet50
	int resnet50[CLASSES][CLASSES] ={
	    {298, 2, 0, 0},
	    {0, 306, 0, 0},
	    {0, 0, 405, 0},
	    {1, 1, 0, 298},
	};
	
	Metrics resnet50M = computeMetrics(resnet50, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] ResNet50 [===]\n";
	cout << "Accuracy: " << resnet50M.accuracy << ";\n";
	cout << "Precision: " << resnet50M.precision << ";\n";
	cout << "Recall: " << resnet50M.recall << ";\n";
	cout << "F1-Score: " << resnet50M.f1 << ".\n\n";


	//ResNet101
	int resnet101[CLASSES][CLASSES] ={
	    {299, 1, 0, 0},
	    {1, 304, 1, 0},
	    {0, 0, 405, 0},
	    {1, 1, 0, 298},
	};
	
	Metrics resnet101M = computeMetrics(resnet101, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] ResNet101 [===]\n";
	cout << "Accuracy: " << resnet101M.accuracy << ";\n";
	cout << "Precision: " << resnet101M.precision << ";\n";
	cout << "Recall: " << resnet101M.recall << ";\n";
	cout << "F1-Score: " << resnet101M.f1 << ".\n\n";


	//ResNet152
	int resnet152[CLASSES][CLASSES] ={
	    {298, 2, 0, 0},
	    {0, 306, 0, 0},
	    {0, 0, 405, 0},
	    {0, 1, 0, 299},
	};
	
	Metrics resnet152M = computeMetrics(resnet152, CLASSES);
	
	cout << setprecision(9);
	cout << "[===] ResNet152 [===]\n";
	cout << "Accuracy: " << resnet152M.accuracy << ";\n";
	cout << "Precision: " << resnet152M.precision << ";\n";
	cout << "Recall: " << resnet152M.recall << ";\n";
	cout << "F1-Score: " << resnet152M.f1 << ".\n\n";


	return 0;
}

Metrics computeMetrics(const int conf[CLASSES][CLASSES], int classes) {
    Metrics m{0, 0, 0, 0};
    
    long double total = 0.0L;
    long double correct = 0.0L;
    
    for(int i  = 0; i < classes; i++) {
        for(int j = 0; j < classes; j++){
            total += conf[i][j];
            if(i == j) correct += conf[i][j];
        }
    }
    
    m.accuracy = (total > 0.0L) ? (static_cast<long double>(correct) / static_cast<long double>(total)) : 0.0L;
    
    long double sum_precision = 0.0L;
    long double sum_recall    = 0.0L;
    long double sum_f1        = 0.0L;
    
    for(int i = 0; i < classes; i++) {
        int TP = conf[i][i];
        int FP = 0;
        int FN = 0;
        
        for(int j = 0; j < classes; j++) {
            if(j != i) {
                FN += conf[i][j];
                FP += conf[j][i];
            }
        }
        
        long double precision = (TP + FP) > 0
            ? (long double)TP / (TP + FP)
            : 0.0L;
            
        long double recall = (TP + FN) > 0
            ? (long double)TP / (TP + FN)
            : 0.0L;
            
        long double f1 = (precision + recall) > 0
            ? 2.0L * precision * recall / (precision + recall)
            : 0.0L;
            
        sum_precision += precision;
        sum_recall += recall;
        sum_f1 += f1;
    }
    
    m.precision = sum_precision / classes;
    m.recall = sum_recall / classes;
    m.f1 = sum_f1 / classes;
    
    return m;
}
