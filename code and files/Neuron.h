//
//  Neuron.h
//  Neural_Assignment_1_Take_3
//
//  Created by MacBook Pro on 10/10/15.
//  Copyright (c) 2015 Neural. All rights reserved.
//

#ifndef Neural_Assignment_1_Take_3_Neuron_h
#define Neural_Assignment_1_Take_3_Neuron_h

#include <vector>
#include <cmath>
using namespace std;
class Neuron
{
    
public:
    Neuron();
    void addEdge(double,double);
    void setEdge(int, double,double);
    void updateWeight(int, double);
    double getInput(int idx);
    double getWeight(int idx);
    int getEdgeCnt();
    double getOutput();
    vector<double> getInputs();
    vector<double> getWeights();
    double actFunction(double);
    double fireNeuron();
    void setGradient_e(double);
    double getGradient_e();
    double getPrevDeltaW (int);
    
    
private:
    vector<double> inputs;
    vector<double> weights;
    double output;
    double gradient_e;
    void addInput(double);
    void addWeight(double);
    
    vector<double> prevDeltaW;
};

Neuron :: Neuron ()
{
    
}

void Neuron :: addEdge (double inp, double w)
{
    addInput(inp);
    addWeight(w);
}

void Neuron :: setEdge (int idx, double inp, double w)
{
    inputs[idx] = inp;
    weights[idx] = w;
}

void Neuron :: updateWeight (int idx, double deltaW)
{
    weights[idx] += deltaW;
    prevDeltaW[idx] = deltaW;
}

double Neuron :: getInput (int idx)
{
    return inputs[idx];
}

double Neuron :: getWeight (int idx)
{
    return weights[idx];
}

double Neuron:: getPrevDeltaW (int idx)
{
    return prevDeltaW[idx];
}

int Neuron :: getEdgeCnt ()
{
    return inputs.size();
}

double Neuron :: getOutput ()
{
    return output;
}

vector<double> Neuron :: getInputs()
{
    return inputs;
}

vector <double> Neuron :: getWeights()
{
    return weights;
}

double Neuron :: actFunction (double inp)
{
    output = tanh (inp);
    return output;
}

double Neuron :: fireNeuron ()
{
    double sum = 0;
    for (int i=0; i<weights.size(); i++)
    {
        sum += (weights[i] * inputs[i]);
    }
    actFunction(sum);
    return output;
}

void Neuron ::  setGradient_e (double inp)
{
    gradient_e = inp;
}

double Neuron :: getGradient_e ()
{
    return gradient_e;
}

void Neuron :: addInput(double inp)
{
    inputs.push_back(inp);
}

void Neuron :: addWeight (double w)
{
    weights.push_back(w);
    prevDeltaW.push_back(0);
}





#endif
