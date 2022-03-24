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

        public void Train(int epochCount, double eta)
        {
            for (int i = 0; i < epochCount; i++)
            {
                TrainNetwork(eta, out var error);
                Console.WriteLine($"\rProgress: {i}/{epochCount}.\tDeviation: {error}");
            }
            
            for (int i = 0; i < _inputs.GetLength(0); i++)
            {
                var aboba = _net.GetResult(_inputs[i]);
                Console.WriteLine($"For input [{string.Join("; ", _inputs[i])}]    got [{string.Join("; ", _net.Layers[^1].Neurons.Select(n => n._value))}]    expected[{string.Join("; ", _examples[i])}]");
            }
        }

        private void TrainNetwork(double eta, out double trainingError)
        {
            trainingError = 0d;
            for (int exampleIndex = 0; exampleIndex < _examples.GetLength(0); exampleIndex++)
            {
                //bake the network (feed forward the appropriate input)
                _net.GetResult(_inputs[exampleIndex]);
                
                //Adjust output layer
                var nodesWeightsDeltas = new double[_net.Layers[^1].NeuronCount];
                for (int nIdx = 0; nIdx < _net.Layers[^1].NeuronCount; nIdx++)
                {
                    var neuronValue = _net.Layers[^1].Neurons[nIdx]._value;
                    var error = neuronValue - _examples[exampleIndex][nIdx];
                    var weightsDelta = error * _net.Layers[^1]._layerFunctions.Derivative(neuronValue);
                    
                    //Store the weights delta associated with current node
                    nodesWeightsDeltas[nIdx] = weightsDelta;
                    
                    //Update the weights of current node
                    for (int wIdx = 0; wIdx < _net.Layers[^1].Neurons[nIdx]._weights.Length; wIdx++)
                    {
                        var currentWeight = _net.Layers[^1].Neurons[nIdx]._weights[wIdx];
                        var prevOutputToWeight = _net.Layers[^2].Neurons[wIdx]._value;
                        var newWeight = currentWeight - prevOutputToWeight * weightsDelta * eta;
                        _net.Layers[^1].Neurons[nIdx].SetWeight(wIdx, newWeight);
                    }
                }
                
                //Adjust hidden layers
                for (int layerIndex = 1; layerIndex < _net.Layers.Count-1; layerIndex++)
                {
                    var lIdx = ^(layerIndex + 1);
                    var currentNodesWeightsDeltas = new double[_net.Layers[lIdx].NeuronCount];
                    
                    for (int nIdx = 0; nIdx < _net.Layers[lIdx].NeuronCount; nIdx++)
                    {
                        var neuronValue = _net.Layers[lIdx].Neurons[nIdx]._value;
                        var error = GetNodeError(nIdx, layerIndex, ref nodesWeightsDeltas);
                        var weightsDelta = error * _net.Layers[lIdx]._layerFunctions.Derivative(neuronValue);
                        
                        //Store the weights delta associated with current node in hidden layer
                        currentNodesWeightsDeltas[nIdx] = weightsDelta;
                        
                        //Update the weights of current hidden layer node
                        for (int wIdx = 0; wIdx < _net.Layers[lIdx].Neurons[nIdx]._weights.Length; wIdx++)
                        {
                            var currentWeight = _net.Layers[lIdx].Neurons[nIdx]._weights[wIdx];
                            var prevOutputToWeight = _net.Layers[^(layerIndex+2)].Neurons[wIdx]._value;
                            var newWeight = currentWeight - prevOutputToWeight * weightsDelta * eta;
                            _net.Layers[lIdx].Neurons[nIdx].SetWeight(wIdx, newWeight);
                        }
                    }
                    
                    //Update nodes weights deltas to be of the adjusted layer
                    nodesWeightsDeltas = currentNodesWeightsDeltas;
                }
                
                //Adjust first layer
                for (int nIdx = 0; nIdx < _net.Layers[0].NeuronCount; nIdx++)
                {
                    var neuronValue = _net.Layers[0].Neurons[nIdx]._value;
                    var error = GetNodeError(nIdx, layerIndex: _net.Layers.Count - 1, ref nodesWeightsDeltas);
                    var weightsDelta = error * _net.Layers[0]._layerFunctions.Derivative(neuronValue);

                    for (int wIdx = 0; wIdx < _net.Layers[0].Neurons[nIdx]._weights.Length; wIdx++)
                    {
                        var currentWeight = _net.Layers[0].Neurons[nIdx]._weights[wIdx];
                        var prevOutputToWeight = _inputs[exampleIndex][wIdx];
                        var newWeight = currentWeight - prevOutputToWeight * weightsDelta * eta;
                        _net.Layers[0].Neurons[nIdx].SetWeight(wIdx, newWeight);
                    }
                }

                trainingError += _net.Layers[^1].Neurons
                    .Select((n, i) => (n._value - _examples[exampleIndex][i]) * (n._value - _examples[exampleIndex][i]))
                    .Sum();
            }
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