
//
// Disclamer:
// ----------
// This code will work only if you have SFML installed and running for this project.
// This code will work only if you selected window, graphics and audio.
//
// Note that the "Run Script" build phase will copy the required frameworks
// or dylibs to your application bundle so you can execute it on any OS X
// computer.
//
// Your resource files (images, sounds, fonts, ...) are also copied to your
// application bundle. To get the path to these resource, use the helper
// method resourcePath() from ResourcePath.hpp
//


//This code will work only if you modify paths for any .txt --- relative paths don't work on OSX

#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>

// Here is a small helper for you ! Have a look.
#include "ResourcePath.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include "Network.h"
using namespace std;


void testShape();

sf::Texture texture;
sf::Sprite sprite;
sf::Image image;


string inFiles [4] = {"line.txt", "wave.txt", "circle.txt", "spiral.txt"};
string outFiles[4] = {"outLine.txt", "outWave.txt", "outCircle.txt", "outSpiral.txt"};
string graphFiles[4] = {"EpochVSMSEGraphLine.txt","EpochVSMSEGraphWave.txt","EpochVSMSEGraphCircle.txt","EpochVSMSEGraphSpiral.txt"};

enum TYPE  {LINE, WAVE, CIRCLE, SPIRAL};

int numLayers;
vector<int> layersDescription;
ifstream IN;
ofstream OUT;
string graphPath;
TYPE shapeType;

double EETA, targetMSE;
long long maxEpochCnt;




int main(int, char const**)
{
    
    
    // Create the main window
    sf::RenderWindow window(sf::VideoMode(400, 400), "SFML window");

    // Set the Icon
    sf::Image icon;
    if (!icon.loadFromFile(resourcePath() + "icon.png")) {
        return EXIT_FAILURE;
    }
    window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());

    
    image.create(400, 400, sf::Color ::White);
    
    srand(time(NULL));
    
    
    shapeType = LINE;
    
    numLayers = 2;
    layersDescription.push_back(10);
    layersDescription.push_back(1);
    
    EETA = 0.01;
    targetMSE = 0.003;
    maxEpochCnt = 100000;
    
    
    
    
    IN.open("/Users/Home/AUC/Fall 15/Neural/Assignments/Neural Assignment 1/Neural Assignment 1/" + inFiles[shapeType]);
    OUT.open("/Users/Home/AUC/Fall 15/Neural/Assignments/Neural Assignment 1/Neural Assignment 1/" + outFiles[shapeType]);
    graphPath = "/Users/Home/AUC/Fall 15/Neural/Assignments/Neural Assignment 1/Neural Assignment 1/" + graphFiles[shapeType];
    
    
    
    
    testShape();
    // Start the game loop
    while (window.isOpen())
    {
        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // Escape pressed: exit
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
                window.close();
            }
        }

        // Clear screen
        window.clear();
        
        texture.loadFromImage(image);
        sprite.setTexture(texture);
        
        window.draw(sprite);

        // Update the window
        window.display();
    }

    return EXIT_SUCCESS;
}



void testShape ()
{
    
    string dummy;
    int singlePatternSize;
    int patternsCnt;
    double x,y,targ;
    double successRate = 0;
    
    double CCR1 = 0;
    double CCR2 = 0;
    
    int CCR1Cnt = 0;
    int CCR2Cnt = 0;
    
    getline(IN, dummy);
    getline(IN, dummy);
    IN >> dummy >> dummy;
    IN >> singlePatternSize;
    IN.ignore();
    getline(IN, dummy);
    IN >> dummy >> dummy;
    IN >> patternsCnt;
    IN.ignore();
    getline(IN, dummy);
    
    
    // 2 == -1 .... 1 == 1
    layersDescription[0] = 10;
    layersDescription[1] = 1;
    
    vector<vector<double>> inputs(patternsCnt);
    
    vector<double> targets(patternsCnt);
    
    for (int i=0; i<patternsCnt; i++)
    {
        IN >> x >> y >> targ;
        inputs[i].push_back(x);
        inputs[i].push_back(y);
        inputs[i].push_back(1);
        
        if (targ==2)
            targets[i] = -1;
        else targets[i] = 1;
    }
    
    Network network (numLayers, layersDescription, inputs, targets);
    network.train(EETA, targetMSE,maxEpochCnt,graphPath);
    OUT << endl << "--------------" << endl;
    for (int i=0; i<inputs.size(); i++)
    {
        double exp = targets[i];
        double test = network.test(inputs[i]);
        OUT << "POINT: " << "(" << inputs[i][0]  <<',' << inputs[i][1] <<"): " << endl;
        OUT << "EXPECTED: " << exp << " -- OUTPUT: " << test << endl;
        OUT << "--------------" << endl;
        
        CCR1Cnt+=(exp<=0);
        CCR2Cnt+=(exp>0);
        
        if (exp<0 && test<0){
            CCR1++;
        }
        else if (exp>0 && test>0){
            CCR2++;
        }
        
        
    }
    OUT << "--------------------" << endl;
    OUT << "--------------------" << endl;
    OUT << "--------------------" << endl;
    OUT << "SUCCESS RATE: " << endl;
    OUT << (CCR1/CCR1Cnt) * 100 <<"---" << (CCR2/CCR2Cnt) * 100<< endl;
    
    
    // test 400*400 points and draw them
    vector<double> testInput(3);
    for (int i=0; i<400; i++)
    {
        for (int j=0; j<400; j++)
        {
            testInput[0] = (i/200.0)-1;
            testInput[1] = (j/200.0)-1;
            testInput[2] = 1;
            
            double test = network.test(testInput);
            if (test<0) image.setPixel(i, j, sf::Color::Red);
            else image.setPixel(i, j, sf::Color::Blue);
        }
    }
    
    
}



