using System;
using System.Linq;
using System.Numerics;
using NeuralNetworks.LayerWorks;
using NeuralNetworks.NetWorks.LayeredNetwork;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.Trainers
{
    public class LayeredNetTrainer
    {
        private readonly LayeredNeuralNet _net;
        private readonly double[][] _examples;
        private readonly double[][] _inputs;

        public LayeredNetTrainer(LayeredNeuralNet net, (double[][] results, double[][] inputs) data)
        {
            _net = net;
            _examples = data.results;
            _inputs = data.inputs;
        }

        public void Train(int epochCount, double learningRate)
        {
            var emptyLine = new string(Enumerable.Range(1, Console.BufferWidth).Select(s => ' ').ToArray());
            for (int i = 0; i < epochCount; i++)
            {
                TrainNetwork(learningRate, out var error);
                
                Console.WriteLine($"\rProgress: {i}/{epochCount}.\tDeviation: {error}");
            }
            
            for (int i = 0; i < _inputs.GetLength(0); i++)
            {
                var aboba = _net.GetResult(_inputs[i]);
                Console.WriteLine($"For input [{string.Join("; ", _inputs[i])}]    got [{string.Join("; ", aboba)}]    expected[{string.Join("; ", _examples[i])}]");
            }
        }

        private void TrainNetwork(double learningRate, out double trainingError)
        {
            trainingError = 0d;
            for (int exampleIdx = 0; exampleIdx < _examples.GetLength(0); exampleIdx++)
            {
                //Feed forward training inputs
                _net.GetResult(_inputs[exampleIdx]);
                BackPropagation(exampleIdx, learningRate);

                trainingError += _net.Layers[^1].Neurons
                    .Select((n, i) => Math.Pow(n._value - _examples[exampleIdx][i], 2))
                    .Sum();
            }
        }

        private void BackPropagation(int exampleIdx, double learningRate)
        {
            var error = _examples[exampleIdx].Select((e, i) => _net.Output[i] - e).ToArray();

            for (int i = _net.Layers.Count-1; i >= 0; i--)
            {
                var outputs = i == 0 ?
                    _inputs[exampleIdx] 
                    : _net.Layers[i - 1].Neurons.Select(n => n._value).ToArray();
                
                error = GetLayerError(_net.Layers[i], error, outputs, learningRate);
            }
        }

        private double[] GetLayerError(Layer currentLayer, double[] nodesError, double[] prevLayerOutputs, double learningRate)
        {
            var nodesDeltaWeight = new double[nodesError.Length];
            var outputErrors = new double[prevLayerOutputs.Length];

            for (int nIdx = 0; nIdx < currentLayer.NeuronCount; nIdx++)
            {
                var currentNode = currentLayer.Neurons[nIdx];
                
                var error = nodesError[nIdx];
                nodesDeltaWeight[nIdx] = error * currentLayer._layerFunctions.Derivative(currentNode._value);

                for (int wIdx = 0; wIdx < currentNode._weights.Length; wIdx++)
                {
                    currentNode._weights[wIdx] -= prevLayerOutputs[wIdx] * nodesDeltaWeight[nIdx] * learningRate;
                    outputErrors[wIdx] += nodesDeltaWeight[nIdx] * currentNode._weights[wIdx];
                }
            }

            return outputErrors;
        }

        private double GetNodeError(int nIdx, int layerIndex, ref double[] nodesWeightsDeltas)
        {
            var res = 0d;
            for (int i = 0; i < _net.Layers[^layerIndex].NeuronCount; i++)
            {
                res += _net.Layers[^layerIndex].Neurons[i]._weights[nIdx] * nodesWeightsDeltas[i];
            }
            return res;
        }
    }
}