// See https://aka.ms/new-console-template for more information
using System;
using NeuralNetworks;
using NeuralNetworks.NetWorks.LayeredNetwork;
using NeuralNetworks.NeuronWorks;
using NeuralNetworks.Trainers;

Console.WriteLine("Hello, World!");

// var aboba = new Neuron(1, NeuronFunctions.Activators.Linear);
//
// // var inputs = new double[][] { new[]{1d}, new[]{2d}};
// // var results = new double[] {1d, 2d};
// //
// // var data = (results, inputs);
// //
// // var trainer = new NeuronTrainer(aboba, data);
// // trainer.Train(100, 0.1);

// var inputs = new double[][] {new[] {1d, 2, 3}, new[] {2d, 3, 4}};
// var results = new double[][] {new[] {0.5d, 0.7d}, new[] {0.3d, 0.8d}};
// var data = (results, inputs);
//
// var net = new LayeredNeuralNet(3, new []{3, 2});
// var trainer = new LayeredNetTrainer(net, data);
// trainer.Train(100, 0.1);

// var inputs = new double[][]
// {
//     new[] {0d, 0, 0},
//     new[] {0d, 0, 1},
//     new[] {0d, 1, 0},
//     new[] {0d, 1, 1},
//     new[] {1d, 0, 0},
//     new[] {1d, 0, 1},
//     new[] {1d, 1, 0},
//     new[] {1d, 1, 1}
// };
//
// var results = new double[][]
// {
//     new[] {0d},
//     new[] {1d},
//     new[] {1d},
//     new[] {0d},
//     new[] {1d},
//     new[] {0d},
//     new[] {0d},
//     new[] {1d}
//};

var inputs = new double[][]
{
    new[] {0d, 0},
    new[] {0d, 1},
    new[] {1d, 0},
    new[] {1d, 1}
};
var results = new double[][]
{
    new[] {0d},
    new[] {0d},
    new[] {0d},
    new[] {1d}
};

var data = (results, inputs);
var net = new LayeredNeuralNet(2, new []{ 128, 25, 1});
var trainer = new LayeredNetTrainer(net, data);
trainer.Train(200, 0.05);