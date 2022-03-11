// See https://aka.ms/new-console-template for more information
using System;
using NeuralNetworks;
using NeuralNetworks.NeuronWorks;
using NeuralNetworks.Trainers;

Console.WriteLine("Hello, World!");

var aboba = new Neuron(1, NeuronFunctions.Activators.Linear);

var inputs = new double[][] { new[]{1d}, new[]{2d}};
var results = new double[] {1d, 2d};

var data = (results, inputs);

var trainer = new NeuronTrainer(aboba, data);
trainer.Train(100, 0.1);