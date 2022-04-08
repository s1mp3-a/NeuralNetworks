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
        private double[][] _layersErrors;

        public LayeredNetTrainer(LayeredNeuralNet net, (double[][] inputs, double[][] results) data)
        {
            _net = net;
            _examples = data.results;
            _inputs = data.inputs;
        }

        public void Train(int epochCount, double learningRate)
        {
            for (int i = 0; i < epochCount; i++)
            {
                TrainNetwork(learningRate, out var error);
                
                Console.WriteLine($"\rProgress: {i}/{epochCount}.\tDeviation: {error}");
            }
            
            for (int i = 0; i < _inputs.GetLength(0); i++)
            {
                var netResult = _net.GetResult(_inputs[i]);
                Console.WriteLine($"For input [{string.Join("; ", _inputs[i])}]    got [{string.Join("; ", netResult)}]    expected[{string.Join("; ", _examples[i])}]");
            }
        }

        private void TrainNetwork(double learningRate, out double trainingError)
        {
            trainingError = 0d;
            for (int exampleIdx = 0; exampleIdx < _examples.GetLength(0); exampleIdx++)
            {

                //Feed forward training inputs so that the neurons have computed their outputs (values)
                _net.GetResult(_inputs[exampleIdx]);
                
                //Calculate per layer errors
                CalculateLayersErrors(exampleIdx);
                
                BackPropagation(exampleIdx, learningRate);

                trainingError += _net.Layers[^1].Neurons
                    .Select((n, i) => Math.Pow(n._value - _examples[exampleIdx][i], 2))
                    .Sum();
            }
        }

        private void CalculateLayersErrors(int exampleIdx)
        {
            _layersErrors = new double[_net.Layers.Count][];

            _layersErrors[^1] = new double[_net.Layers[^1].NeuronCount];
            for (int neuronIdx = 0; neuronIdx < _net.Layers[^1].Neurons.Length; neuronIdx++)
            {
                _layersErrors[^1][neuronIdx] = _examples[exampleIdx][neuronIdx] - _net.Layers[^1].Neurons[neuronIdx]._value;
            }
            
            for (int layerIdx = _net.Layers.Count - 2; layerIdx >= 0; layerIdx--)
            {
                var leftLayer = _net.Layers[layerIdx];
                var rightLayer = _net.Layers[layerIdx + 1];
                _layersErrors[layerIdx] = new double[leftLayer.NeuronCount];
                
                for (int neuronIdx = 0; neuronIdx < leftLayer.NeuronCount; neuronIdx++)
                {
                    var nodeError = 0d;
                    for (int rightNIdx = 0; rightNIdx < rightLayer.NeuronCount; rightNIdx++)
                    {
                        nodeError += _layersErrors[layerIdx + 1][rightNIdx] * rightLayer.Neurons[rightNIdx]._weights[neuronIdx]; //TODO maybe index misplacement
                    }

                    _layersErrors[layerIdx][neuronIdx] = nodeError;
                }
            }
        }

        private void BackPropagation(int exampleIdx, double learningRate)
        {
            for (int i = _net.Layers.Count-1; i >= 0; i--)
            {
                AdjustWeightsAndBias(i, exampleIdx, learningRate);
            }
        }

        private void AdjustWeightsAndBias(int layerIndex, int exampleIdx, double learningRate)
        {
            var layer = _net.Layers[layerIndex];
            for (int neuronIdx = 0; neuronIdx < layer.Neurons.Length; neuronIdx++)
            {
                var neuron = layer.Neurons[neuronIdx];
                var gradient = _layersErrors[layerIndex][neuronIdx] * layer._layerFunctions.Derivative(neuron._value);
                
                for (int weightIdx = 0; weightIdx < neuron._weights.Length; weightIdx++)
                {
                    double deltaWeight = learningRate
                                         * gradient
                                         * GetPreviousLayerNeuronToWeightOutput(layerIndex, weightIdx, exampleIdx);
                    
                    neuron._weights[weightIdx] = neuron._weights[weightIdx] + deltaWeight;
                }

                neuron._bias = neuron._bias + gradient;
            }
        }

        private double GetPreviousLayerNeuronToWeightOutput(int currentLayerIndex, int weightIdx, int exampleIdx)
        {
            var lIdx = currentLayerIndex - 1;
            return currentLayerIndex == 0 ? _inputs[exampleIdx][weightIdx] : _net.Layers[lIdx].Neurons[weightIdx]._value;
        }
    }
}