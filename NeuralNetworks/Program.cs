// See https://aka.ms/new-console-template for more information

using System;
using NeuralNetworks;
using NeuralNetworks.DataExport;
using NeuralNetworks.NetWorks.LayeredNetwork;
using NeuralNetworks.NetWorks.RbNetwork;
using NeuralNetworks.NeuronWorks;
using NeuralNetworks.Trainers;

LayeredNNTrainingData.Data testData = LayeredNNTrainingData.SQRT_X;
//
// var functions = new[]
// {
//     NeuronFunctions.Functions.Linear,
//     NeuronFunctions.Functions.Sigmoid
// };
//
//var testInputPMaboba = Enumerable.Range(0, 25).Select(x => 1d).ToArray();
//
var data = (testData.inputs, testData.results);
// var net = new LayeredNeuralNet(testData.inputLayerNeuronCount, new[] {25, testData.outputLayerNeuronCount},
//     functions);
// var trainer = new LayeredNetTrainer(net, data);
// trainer.Train(100, 0.003);

var rbNet = new RbNetwork(data, s: 3);
rbNet.Train(10000, 0.05);

var rbNetInputs = Enumerable.Range(0, 80).Select((x, i) => new double[] {(double) x / 3}).ToArray();
var rbNetResults = new double[rbNetInputs.GetLength(0)];

for (int i = 0; i < rbNetInputs.GetLength(0); i++)
{
    rbNetResults[i] = rbNet.GetResult(rbNetInputs[i])[0];
}

CsvExporter.ExportCsv("/Users/s1mpl3/Desktop/in.txt", rbNetInputs.Select(x => x[0]).ToArray());
CsvExporter.ExportCsv("/Users/s1mpl3/Desktop/out.txt", rbNetResults);