// See https://aka.ms/new-console-template for more information
using System;
using NeuralNetworks;
using NeuralNetworks.NetWorks.LayeredNetwork;
using NeuralNetworks.NeuronWorks;
using NeuralNetworks.Trainers;

LayeredNNTrainingData.Data testData = LayeredNNTrainingData.XOR_ABC;

var functions = new[]
{
    NeuronFunctions.Functions.Sigmoid,
    NeuronFunctions.Functions.Linear,
};

var data = (testData.inputs, testData.results);
var net = new LayeredNeuralNet(testData.inputLayerNeuronCount, new []{25, 25, testData.outputLayerNeuronCount});
var trainer = new LayeredNetTrainer(net, data);
trainer.Train(2000, 0.2);