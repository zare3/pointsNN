//
//  Network.h
//  Neural_Assignment_1_Take_3
//
//  Created by MacBook Pro on 10/10/15.
//  Copyright (c) 2015 Neural. All rights reserved.
//

#ifndef Neural_Assignment_1_Take_3_Network_h
#define Neural_Assignment_1_Take_3_Network_h

#include "Neuron.h"
#include <fstream>
#include <ctime>


class Network
{
    
    
public:
    Network (int, vector<int>, vector<vector<double> > , vector<double>);
    void train (double,double,int,string);
    double test (vector<double>);
    
    
private:
    int numLayers;
    vector<int> layersDescription;
    vector<double> targets;
    vector<vector<double> > inputs;
    vector<vector<double> > testingInput;
    vector<vector<Neuron> > neurons;
    double eeta;
    int targetEpochCnt;
    double targetMSE;
    double currentMSE;
    
    double getRandNumber (int,int);
    void updateMSE(int);
    
    ofstream drawGraph;
    
    
    /* Multi Layer Functions*/
    void initMultiLayer();
    void buildNetwork();
    void trainMultiLayer();
    void forwardPass(int);
    void initInputLayer();
    void backPropagate(int);
    /* ----------------------*/
    
    
    void trainSingleLayer ();
    
};



Network :: Network (int numLayers, vector<int> layersDescription, vector<vector<double> > inputs, vector<double> targets)
{
    srand(time(NULL));
    this->numLayers = numLayers;
    this->layersDescription = layersDescription;
    this->inputs = inputs;
    this->targets = targets;
    
    this->currentMSE = 1e9;
}

void Network :: train(double eeta, double targetMSE, int targetEpochCnt, string drawGraphPath)
{
    this->eeta = eeta;
    this->targetMSE = targetMSE;
    this->targetEpochCnt = targetEpochCnt;
    this->drawGraph.open(drawGraphPath.c_str());;
    srand(time(NULL));
    if (numLayers>1){
        
        initMultiLayer();
        trainMultiLayer();
    }
    else
    {

        initInputLayer();
        trainSingleLayer();
    }

    
}


void Network :: initMultiLayer ()
{
    
    initInputLayer();
    buildNetwork();
    backPropagate(0);
    
}

void Network :: initInputLayer()
{
    neurons.resize(numLayers);
    
    for (int i=0; i<neurons.size(); i++)
    {
        neurons[i].resize(layersDescription[i]);
    }
    
    for (int i=0; i<neurons[0].size(); i++)
    {
        for (int j=0; j<inputs[0].size(); j++)
        {
            neurons[0][i].addEdge(inputs[0][j],getRandNumber(0,1));
        }
    }
}

void Network :: buildNetwork ()
{
    
    
    
    for (int i=1; i<neurons.size(); i++)
    {
        for (int j=0; j<neurons[i].size(); j++)
        {
            neurons[i][j].addEdge(1, getRandNumber(0, 1));
        }
    }
    
    
    // propagate output from each layer to make output layer ready
    for (int i=0; i<neurons.size()-1; i++)
    {
        for (int j=0; j<neurons[i].size(); j++)
        {
            double neuronOut = neurons[i][j].fireNeuron();
            for (int k=0; k<neurons[i+1].size(); k++)
            {
                neurons[i+1][k].addEdge(neuronOut,getRandNumber(0,1));
            }
            
        }
    }
    
    // fire output layer
    for (int i=0; i<neurons[neurons.size()-1].size(); i++)
    {
        neurons[neurons.size()-1][i].fireNeuron();
    }
    
    
}

void Network :: backPropagate (int patternIdx)
{
    
    // output layer
    for (int i=0; i<neurons[neurons.size()-1].size(); i++)
    {
        for (int j=0; j< neurons[neurons.size()-1][i].getEdgeCnt(); j++)
        {
            double aJ = neurons[neurons.size()-2][j].getOutput();
            double Ti = targets[patternIdx];
            double Oi = neurons[neurons.size()-1][i].getOutput();
            double gradient_e =   (Ti - Oi)  * (1 - Oi * Oi);
            neurons[neurons.size()-1][i].setGradient_e(gradient_e);
            double deltaW = eeta * aJ * gradient_e;
            neurons[neurons.size()-1][i].updateWeight(j,deltaW);
        }
    }
    
    // back_propagate
    for (int i=neurons.size()-2; i>=0; i--) // each layer
    {
        for (int j=0; j<neurons[i].size(); j++) // each node
        {
            double aJ = neurons[i][j].getOutput();
            int cnt = neurons[i][j].getEdgeCnt();
            double summation = 0;
            
            
            for (int l=0; l<neurons[i+1].size(); l++)
            {
                summation += (neurons[i+1][l].getWeight(j) * neurons[i+1][l].getGradient_e());
            }
            neurons[i][j].setGradient_e((1-aJ * aJ) * summation);
            
            for (int k=0; k<cnt; k++) // each edge
            {
                double Ik = neurons[i][j].getInput(k);
                double deltaW;
                deltaW = eeta * Ik * neurons[i][j].getGradient_e();
                if ((deltaW < 0 && neurons[i][j].getPrevDeltaW(k) < 0) || (deltaW > 0 && neurons[i][j].getPrevDeltaW(k) > 0)) deltaW*=1.2;
                else deltaW*=0.5;
                neurons[i][j].updateWeight(k,deltaW);
            }
            
        }
    }
}



void Network :: forwardPass (int patternIdx)
{
    
    
    // first layer inputs
    for (int i=0; i<neurons[0].size(); i++)
    {
        for (int j=0; j<inputs[patternIdx].size(); j++)
        {
            neurons[0][i].setEdge(j,inputs[patternIdx][j],neurons[0][i].getWeight(j));
        }
    }
    // compute forward pass till last layer
    for (int i=0; i<neurons.size()-1; i++)
    {
        for (int j=0; j<neurons[i].size(); j++)
        {
            double neuronOut = neurons[i][j].fireNeuron();
            for (int k=0; k<neurons[i+1].size(); k++)
            {
                neurons[i+1][k].setEdge(j,neuronOut, neurons[i+1][k].getWeight(j));
            }
        }
    }
    
    // fire output layer
    for (int i=0; i<neurons[neurons.size()-1].size(); i++)
    {
        neurons[neurons.size()-1][i].fireNeuron();
    }
    updateMSE(patternIdx);
}

double Network:: getRandNumber(int LO, int HI)
{
    return(LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
}

void Network :: updateMSE(int patternIdx)
{
    double sum = currentMSE;
    if (sum == 1e9) sum = 0;
    for (int i=0; i<neurons[neurons.size()-1].size(); i++)
    {
        double output = neurons[neurons.size()-1][i].getOutput();
        double target = targets[patternIdx];
        sum += (( output - target ) * (output - target) );
    }
    currentMSE = sum;
    
}

void Network :: trainMultiLayer()
{
    int epochCnt = 0;
    int patternIdx = 0;
    bool reset = 0;
    while (1)
    {
        if (reset)
        {
            currentMSE = 0;
            reset = 0;
        }
        forwardPass(patternIdx);
        backPropagate(patternIdx);
        patternIdx++;
        
        patternIdx = patternIdx % inputs.size();
        if (patternIdx == 0 )
        {
            epochCnt++;
            currentMSE/=inputs.size();
            //cout << currentMSE << endl;
            reset = 1;
            drawGraph << "EPOCH: " << epochCnt << "----- MSE: " << currentMSE << endl;
            cout << "EPOCH: " << epochCnt << "----- MSE: " << currentMSE << endl;
            if ( currentMSE < targetMSE || epochCnt > targetEpochCnt){
                break;
            }
        }
        
        
    }
    this->drawGraph.close();
    
    
}



double Network :: test (vector<double> testInput)
{
    // first layer inputs
    for (int i=0; i<neurons[0].size(); i++)
    {
        for (int j=0; j<testInput.size(); j++)
        {
            neurons[0][i].setEdge(j,testInput[j],neurons[0][i].getWeight(j));
        }
    }
    
    // compute forward pass till last layer
    for (int i=0; i<neurons.size()-1; i++)
    {
        for (int j=0; j<neurons[i].size(); j++)
        {
            double neuronOut = neurons[i][j].fireNeuron();
            for (int k=0; k<neurons[i+1].size(); k++)
            {
                neurons[i+1][k].setEdge(j,neuronOut, neurons[i+1][k].getWeight(j));
            }
        }
    }
    
    // fire output layer
    double res;
    for (int i=0; i<neurons[neurons.size()-1].size(); i++)
    {
        res = neurons[neurons.size()-1][i].fireNeuron();
    }
    return res;
}





void Network::trainSingleLayer()
{
    srand(time(NULL));
    
    int patternIdx = 0;
    int epochCnt = 0;
    while (currentMSE > targetMSE || currentMSE==0 )
    {
        
        // set layer inputs
        for (int i=0; i<neurons[0].size(); i++)
        {
            for (int j=0; j<inputs[patternIdx].size(); j++)
            {
                neurons[0][i].setEdge(j,inputs[patternIdx][j],neurons[0][i].getWeight(j));
            }
        }
        int i,j;
        for ( i=0; i<neurons.size(); i++)
        {
            for (j=0; j<neurons[i].size(); j++)
            {
                int cnt = neurons[i][j].getEdgeCnt();
                double deltaW;
                double output = neurons[i][j].fireNeuron();
                for (int k=0; k<cnt; k++)
                {
                    deltaW = eeta * (targets[patternIdx] - output ) * neurons[i][j].getInput(k);
                    neurons[i][j].updateWeight(k, deltaW);
                }
            }
        }
        updateMSE(patternIdx);
        patternIdx++;
        patternIdx%=inputs.size();
        
        if (patternIdx == 0)
        {
            
            epochCnt++;
            currentMSE/=inputs.size();
            drawGraph << "EPOCH: " << epochCnt << "----- MSE: " << currentMSE << endl;
            cout << "EPOCH: " << epochCnt << "----- MSE: " << currentMSE << endl;
            if (epochCnt > targetEpochCnt) break;
            currentMSE = 0;
            
        }
    }
    
}

#endif
